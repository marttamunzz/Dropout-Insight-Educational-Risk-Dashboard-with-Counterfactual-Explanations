import shutil
import base64
import io
import os
import webbrowser
import threading
import pickle
import warnings
import hashlib, json, sys, tempfile

import dash_bootstrap_components as dbc
import matplotlib
import pandas as pd
import numpy as np
import shap

from app import app
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from pathlib import Path

from supervised.automl import AutoML
from tabs.AutoMLReportTab import AutoMLReportTab
from tabs.FeaturesImportancesTab import FeaturesImportanceBasicTab, FeaturesImportanceExpertTab
from tabs.ClassificationStatsTab import ClassificationStatsTab
from tabs.WhatIfTab import WhatIfBasicTab, WhatIfExpertTab
from tabs.CounterfactualsTab import CounterfactualsTab
from tabs.GroupCounterFactualsTab import GroupCounterfactualsTab
from waitress import serve

matplotlib.use('agg')

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
css_path = [
    os.path.join(BASE_DIR, "assets", "fonts.css"),
    os.path.join(BASE_DIR, "assets", "custom.css"),
]

print("CSS path passed to explainerdashboard:")
for p in css_path:
    print("  ", p, "->", os.path.exists(p))

#NUEVO CONTENIDO PARA GUARDAR LOS SHAP A VER SI NO TENGO QUE CARGARLOS CADA VEZ QUE SE EJECUTE LA APP

# --- Calcula SHAP una vez dentro del explainer y re-guarda el .dill (atómico) ---
def ensure_shap_inside_explainer(explainer, dill_path: str):
    """
    Si el explainer no tiene SHAP cacheados internamente, los calcula UNA VEZ
    y re-guarda el .dill de forma atómica. Si ya están, no hace nada.
    """
    # ¿ya existen en memoria?
    if getattr(explainer, "_shap_values", None) is not None or hasattr(explainer, "_shap_values_df"):
        print("✅ SHAP ya viven dentro del explainer (no se recalcula).")
        return

    print("🟡 SHAP no están dentro. Calculando una vez desde el explainer...")
    if hasattr(explainer, "get_shap_values_df"):
        _ = explainer.get_shap_values_df()
    elif hasattr(explainer, "get_shap_values"):
        _ = explainer.get_shap_values()
    else:
        _ = explainer.shap_values

    # ⬇️ usa un temporal con extensión .dill para forzar dill
    tmp = dill_path + ".tmp.dill"
    explainer.dump(tmp)          
    os.replace(tmp, dill_path)  
    print(f"💾 Explainer re-guardado con SHAP dentro: {dill_path}")

def csv_sha256_from_contents(contents: str) -> str:
    _, b64 = contents.split(',', 1)
    return hashlib.sha256(base64.b64decode(b64)).hexdigest()

def atomic_save_npz(path: Path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as tmp:
        np.savez_compressed(tmp.name, **arrays)
        tmp.flush(); os.fsync(tmp.fileno())
        tmpname = tmp.name
    os.replace(tmpname, path)

def build_model_fingerprint(model) -> str:
    rid = getattr(model, "random_state", None)
    return f"{type(model).__name__}:{rid}"

def file_sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def _perm_noop(*a, **k):
    return None

def disable_permutation(explainer):
    # bandera que consulta explainerdashboard
    try:
        explainer.permutation_importance = False
    except Exception:
        pass
    # métodos que podrían disparar el cálculo en algunas versiones
    for name in (
        "get_perm_importances_df",
        "get_permutation_importances_df",
        "calc_permutation_importances",
        "calculate_permutation_importances",
    ):
        if hasattr(explainer, name):
            setattr(explainer, name, _perm_noop)


def inject_cached_shap_into_explainer(explainer, csv_hash: str, shap_cache_dir: Path, explainer_dill_path: str, proba_idx: int = 1):
    """
    Carga SHAP desde .npz y los asigna al explainer (RAM).
    Si la caché es antigua (dtype=object o pickled), la purga y la reconstruye.
    Si no hay caché, calcula una única vez con salida 1D (proba[:, proba_idx]) y guarda float32.
    NO reescribe el .dill.
    """
    # --- Normaliza X ---
    explainer.X.index = explainer.X.index.map(str)
    explainer.X.sort_index(inplace=True)

    model_fp = file_sha256(explainer_dill_path)[:16]   # huella estable del explainer guardado
    key = hashlib.sha256(
        f"{csv_hash}|{model_fp}|{tuple(explainer.X.columns)}|{explainer.X.shape}".encode()
    ).hexdigest()[:24]

    shap_cache_dir.mkdir(parents=True, exist_ok=True)
    shap_npz = shap_cache_dir / f"shap_{key}.npz"
    shap_meta = shap_cache_dir / f"shap_{key}.json"

    def _assign(values, base_values, data, feature_names):
        explainer._shap_values = values
        try:
            explainer._shap_base_value = base_values
        except Exception:
            pass
        # Alinear X con los datos guardados si están
        if data is not None and data.shape[0] == explainer.X.shape[0]:
            explainer.X = pd.DataFrame(data, columns=feature_names, index=explainer.X.index)
        # Reset caches de dataframes si existen
        if hasattr(explainer, "_shap_values_df"):
            explainer._shap_values_df = None
        if hasattr(explainer, "_shap_interaction_values_df"):
            explainer._shap_interaction_values_df = None

    # --- Intentar leer caché existente ---
    def _try_load_cache() -> bool:
        if not (shap_npz.exists() and shap_meta.exists()):
            return False
        try:
            npz = np.load(shap_npz, allow_pickle=False)
        except ValueError:
            # Caché antigua con objetos -> purgar
            print("⚠️ SHAP cache contains pickled/object arrays. Rebuilding...")
            try:
                shap_npz.unlink()
            except: pass
            try:
                shap_meta.unlink()
            except: pass
            return False

        values = npz["values"]
        # Si aún así es object, purgamos
        if values.dtype == object:
            print("⚠️ SHAP cache dtype=object. Rebuilding...")
            try:
                shap_npz.unlink()
            except: pass
            try:
                shap_meta.unlink()
            except: pass
            return False

        base_values = npz["base_values"] if "base_values" in npz and npz["base_values"].size else None
        data = npz["data"] if "data" in npz else None
        meta = json.loads(shap_meta.read_text(encoding="utf-8"))
        feature_names = meta.get("feature_names", list(explainer.X.columns))
        _assign(values, base_values, data, feature_names)
        print(f"✅ SHAP loaded from cache and injected: {shap_npz}")
        return True

    if _try_load_cache():
        return True  # cache hit válido

    # --- No hay caché válida: calcular y guardar en formato numérico 1D ---
    import shap
    # Forzamos salida 1D para evitar multi-salida (que a veces genera dtype=object)
    if hasattr(explainer.model, "predict_proba"):
        def pred_fn(A):
            P = explainer.model.predict_proba(A)
            # seguridad por si hay solo 1 columna
            idx = proba_idx if P.shape[1] > proba_idx else (P.shape[1]-1)
            return P[:, idx]
    else:
        pred_fn = explainer.model.predict

    # Background pequeño y determinista
    bg = explainer.X.sample(n=min(100, len(explainer.X)), random_state=42)
    masker = shap.maskers.Independent(bg)
    sv = shap.Explainer(pred_fn, masker)(explainer.X)

    # Tomamos arrays puros y los pasamos a float32
    raw_values = getattr(sv, "values", sv)
    values = np.asarray(raw_values)
    if values.dtype == object:
        # Intento de apilado si fuese lista de arrays homogénea
        try:
            values = np.stack(list(values), axis=-1)
        except Exception:
            raise RuntimeError(
                "SHAP devolvió dtype=object. Usa salida 1D (prob_clase) ajustando proba_idx."
            )
    values = values.astype(np.float32, copy=False)

    base_values = getattr(sv, "base_values", None)
    if base_values is not None:
        base_values = np.asarray(base_values).astype(np.float32, copy=False)

    # data: usa X numérica
    data = explainer.X.to_numpy(dtype=np.float32, copy=False)

    # Guardado atómico
    atomic_save_npz(
        shap_npz,
        values=values,
        base_values=base_values if base_values is not None else np.array([]),
        data=data
    )

    meta = {
        "csv_hash": csv_hash,
        "model_fp": model_fp,
        "feature_names": list(explainer.X.columns),
        "values_shape": list(values.shape),
        "background_size": int(len(bg)),
    }
    shap_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    _assign(values, base_values, data, meta["feature_names"])
    print(f"💾 SHAP computed once, cached and injected: {shap_npz}")
    return False


#TERMINA EL CONTENIDO

invalid_type_alert = dbc.Alert(
    children="Invalid dataset type. Please be sure it is a .csv file.",
    color="danger",
    dismissable=True,
    duration=10000
)

no_file_alert = dbc.Alert(
    children="Please upload the dataset before pressing Start.",
    color="danger",
    dismissable=True,
    duration=10000
)

app.layout = dbc.Container([
    html.Div([
        dbc.Navbar([
            html.A(
                dbc.Row([
                    dbc.Col(dbc.NavbarBrand(
                        "Dropout Insight: Educational Risk Dashboard with Counterfactual Explanations", className="ms-2"
                    )),
                ], className="g-0 ml-auto flex-nowrap mt-3 mt-md-0", align="center"),
            ),
        ]),

        html.Div(
    id="upload-div",
    children=[
        html.Br(),
        dbc.Card(
            dbc.CardBody([
                html.H2("Welcome 👋"),
                html.P("Follow these steps before uploading your dataset:"),

                # Two-column checklist
                dbc.Row([
                    dbc.Col(
                        dbc.ListGroup([
                            dbc.ListGroupItem("✅ Upload a .csv file"),
                            dbc.ListGroupItem("✅ (Optional) The first column should be the index"),
                            dbc.ListGroupItem("✅ Include at least one continuous (float) column"),
                            dbc.ListGroupItem('✅ Target in the last column with two values: "Dropout" / "No dropout"'),
                        ]),
                        md=7
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("Need a template?"),
                                html.P("Download a minimal CSV structure to get started quickly."),
                                dbc.Button("Download sample CSV", id="btn-sample-csv", color="secondary", size="sm"),
                                dcc.Download(id="dl-sample-csv"),
                            ]),
                            className="shadow-sm"
                        ),
                        md=5
                    ),
                ], className="g-3"),

                html.Hr(),

                html.H4("Load dataset"),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%", "height": "60px", "lineHeight": "60px",
                        "borderWidth": "1px", "borderStyle": "dashed",
                        "borderRadius": "5px", "textAlign": "center", "margin": "10px",
                    },
                    multiple=False,
                ),

                # Live checklist (turns green/red after a file is uploaded)
                html.Div(id="live-checklist", className="mt-2"),
            ])
        ),
    ]
),
        html.Br(),
        dbc.Spinner(dbc.Row(dbc.Col([html.Div(id="output-data")]))),

        html.Br(),
        dbc.Button("Start", color="primary", id="start-button",
                   style={'textAlign': 'center'}),

        html.Br(),
        dbc.Spinner(html.Div(id='dashboard-button', children=dbc.Button(
            "Go to dashboard", id='dashboard-button-link', href="http://127.0.0.1:8050/", target="_blank"
        ), hidden=True)),


        html.Br(),
        html.Div(id='placeholder'),
        html.Br(),
        html.Div(id="alert", children=[]),
    ])
])

# 1) Download sample CSV
@app.callback(
    Output("dl-sample-csv", "data"),
    Input("btn-sample-csv", "n_clicks"),
    prevent_initial_call=True
)
def download_template(n):
    import pandas as pd
    sample = pd.DataFrame({
        "student_id": [1, 2, 3],
        "attendance": [0.85, 0.72, 0.91],
        "avg_grade": [6.1, 5.2, 7.8],
        "Dropout": ["Dropout", "No dropout", "No dropout"]
    })
    return dcc.send_data_frame(sample.to_csv, "sample_dataset.csv", index=False)

# 2) Live checklist after upload
@app.callback(
    Output("live-checklist", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def validate_requirements(contents, filename):
    import pandas as pd
    if contents is None or not filename:
        raise PreventUpdate

    # Uses your existing helper to parse the CSV
    df = parse_data(contents, filename)
    if df is None:
        return dbc.Alert("The file could not be processed. Please check the format.", color="danger")

    is_csv = filename.lower().endswith(".csv")
    #index_ok = df.index.is_unique
    has_float = any(str(t).startswith("float") for t in df.dtypes)
    last_vals = set(map(lambda x: str(x).strip().lower(), df.iloc[:, -1].unique()))
    target_ok = last_vals.issubset({"dropout", "no dropout"}) and len(last_vals) <= 2

    def item(ok, text):
        color = "success" if ok else "danger"
        icon = "✅" if ok else "❌"
        return dbc.ListGroupItem(f"{icon} {text}", color=color, className="py-1")

    return dbc.Card(dbc.CardBody([
        html.H5("Requirements check"),
        dbc.ListGroup([
            item(is_csv, "File is .csv"),
            #item(index_ok, "Index column correctly set"),
            item(has_float, "At least one continuous (float) column"),
            item(target_ok, 'Target in the last column with two values: "Dropout" / "No dropout"'),
        ])
    ]), className="mt-2")



def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # searching for the single column to be the index
            for col in df.columns:
                if df[col].is_unique:
                    df.set_index(col, inplace=True)
                    print(f"✅ Using '{col}' as index (single column detected)")
                    break
            else:
                # If there is no single column, create one artificial
                df.reset_index(drop=True, inplace=True)
                print("⚠️ There is no single column detected. RangeIndex will be used.")

            return df
        else:
            raise ValueError("❌ Wrong file format (only .csv allowed)")
    except Exception as e:
        print(f"❌ Error processing the file: {e}")
        return None

def check_requirements_for_training(df, filename):
    """Validate requirements before training the model."""
    errors = []

    # 1) format of the file
    if not filename.lower().endswith(".csv"):
        errors.append("The file must be a .csv")

    # 2) Target
    last_vals = {str(x).strip().lower() for x in df.iloc[:, -1].unique()}
    if last_vals != {"dropout", "no dropout"}:
        errors.append("The last column must be the Target, if not, the training cannot be done")

    # 3) At least one continious column
    has_float = any(str(t).startswith("float") for t in df.dtypes)
    if not has_float:
        errors.append("Dataset must include at least one continuous (float) column")

    # 4) Missing values
    if df.isnull().any().any():
        errors.append("Dataset must not contain missing values (NaN). Please clean the data before uploading.")


    return errors 


def create_AutoML_model(contents, filename):
    df = parse_data(contents, filename)
    print("Index after set_index:", df.index[:5])
    print("Type of index:", type(df.index))

    df[df.columns[-1]] = df[df.columns[-1]
                            ].replace({"Dropout": 0, "No dropout": 1})
    df["Target"] = df[df.columns[-1]]  # Add the target column

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        X.index, y, test_size=0.20, stratify=y
    )

    X_train = df.loc[X_train_idx].drop(columns=["Target"])
    X_test = df.loc[X_test_idx].drop(columns=["Target"])
    y_train = df.loc[X_train_idx]["Target"]
    y_test = df.loc[X_test_idx]["Target"]

    print("📌 X_test index before explainer:", X_test.index[:5])
    print("📌 Type of index:", type(X_test.index))

    dataset_name = filename.split(".")[0]
    model_path = os.path.join("saved_models", dataset_name, "AutoML")
    report_path = os.path.join("saved_models", dataset_name, "reportML.pkl")

    if os.path.exists(model_path):
        model = AutoML(model_path)

        # Loading reportML.pkl if exists
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                reportML = pickle.load(f)
        else:
            reportML = model.report()
            with open(report_path, "wb") as f:
                pickle.dump(reportML, f)
            print(f"ReportML saved in: {report_path}")

        return model, X_test, y_test, reportML, model, df, True

    os.makedirs(model_path, exist_ok=True)
    model = AutoML(
        results_path=model_path,
        algorithms=["Baseline", "Linear", "Decision Tree", "Random Forest", "Extra Trees",
                    "Xgboost", "LightGBM", "CatBoost", "Neural Network", "Nearest Neighbors"],
        start_random_models=1,
        stack_models=True,
        train_ensemble=True,
        explain_level=2,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.80,
            "shuffle": True,
            "stratify": True,
        })

    trained_model = model.fit(X_train, y_train)
    model_report = model.report()

    # Save reportML with pickle
    with open(report_path, "wb") as f:
        pickle.dump(model_report, f)

    df_features = df[df.columns[:-1]]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(df_features)
    df['cluster'] = kmeans.labels_

    return model, X_test, y_test, model_report, trained_model, df, False


@app.callback(
    Output('output-data', 'children'),
    [Input('upload-data', 'contents'), Input('upload-data', 'filename')],
    prevent_initial_call=True
)
def update_table(contents, filename):
    if contents:
        try:
            df = parse_data(contents, filename)
        except Exception:
            return invalid_type_alert

        table = html.Div([
            html.H4(filename),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict("records"),
                style_table={'overflowX': 'scroll'},
                sort_mode='multi',
                page_action='native',
                page_current=0,
                page_size=20,
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            ),
            html.Hr(),
            html.Div("Raw Content"),
            html.Pre(contents[0:200] + "...",
                     style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"}),
        ])
        return table


@app.callback(
    Output('alert', 'children'),
    Output('placeholder', 'children'),
    Input('start-button', 'n_clicks'),            
    State('upload-data', 'contents'),            
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def create_dashboard(n_clicks, contents, filename):
    global db1, db2

    if not n_clicks:
        raise PreventUpdate
    if contents is None or filename is None:
        return no_file_alert, True

    #OTRA COSA PARA PROBAR QUE OCURRE SI GUARDO LOS SHAP Y SI NO SE TIENEN Q CALCULAR SIEMPRE (SOLO UNA LINEA)
    csv_hash = csv_sha256_from_contents(contents)

    df = parse_data(contents, filename)
    if df is None:
        # Error 
        return invalid_type_alert, True

    # Validation when the user press "start"
    errors = []

    # 1) .csv
    if not filename.lower().endswith(".csv"):
        errors.append("The file must be a .csv.")

    # 2) continious column
    has_float = any(str(t).startswith("float") for t in df.dtypes)
    if not has_float:
        errors.append("Dataset must include at least one continuous (float) column.")

    # 3) Target 
    last_vals = {str(x).strip().lower() for x in df.iloc[:, -1].unique()}
    if last_vals != {"dropout", "no dropout"}:
        errors.append("The last column must be the Target with exactly two values: 'Dropout' / 'No dropout'.")

    # Alerts
    if errors:
        return dbc.Alert(
            children=html.Ul([html.Li(msg) for msg in errors]),
            color="danger",
            dismissable=True,
            duration=10000
        ), True

    # OK
    dataset_name = filename.split(".")[0]
    saved_dir = os.path.join("saved_models", dataset_name)
    saved_explainer_path = os.path.join(saved_dir, "explainer.dill")
    report_path = os.path.join(saved_dir, "reportML.pkl")
    os.makedirs(saved_dir, exist_ok=True)
    print(f"Folder created or existing: {saved_dir}")

    #OTRA COSA PARA PROBAR QUE OCURRE SI GUARDO LOS SHAP Y SI NO SE TIENEN Q CALCULAR SIEMPRE (SOLO UNA LINEA)
    shap_cache_dir = Path(saved_dir) / "shap_cache"


    explainer = None
    trained = None
    reportML = None

    if os.path.exists(saved_explainer_path):
        try:
            print("🔁 Loading explainer from:", saved_explainer_path)
            explainer = ClassifierExplainer.from_file(saved_explainer_path)

            # ¡justo aquí!
            disable_permutation(explainer)

            print("Explainer loaded correctly")
            expected_index = getattr(explainer, "index_backup", None)

            if expected_index is None:
                return dbc.Alert("Explainer without index. Upload the dataset again.", color="danger"), True

            print("🧠 explainer.X.index[:5]:", explainer.X.index[:5])
            print("🧠 index_backup[:5]:", expected_index[:5])

            idx_actual = list(map(str, explainer.X.index))
            idx_backup = list(map(str, expected_index))
            print("🧠 ¿Index = backup? :", idx_actual == idx_backup)


            if idx_actual != idx_backup:
                return dbc.Alert("The explainer is corrupt. Upload the dataset again to regenerate it.", color="danger"), True
            
            ensure_shap_inside_explainer(explainer, saved_explainer_path)


            #MAS COSAS AÑADIDAS PARA NO TENER QUE RECALCULAR LOS SHAP SIEMPREEEEEE 
            if hasattr(explainer, "permutation_importance"):
                explainer.permutation_importance = False

            # Dataset y modelo desde explainer
            df = explainer.X.copy()
            df["Target"] = explainer.y
            trained = explainer.model

            # Cargar reportML si existe
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    reportML = pickle.load(f)
            else:
                print("reportML.pkl not founded")
                reportML = None

            model_path = os.path.join("saved_models", dataset_name, "AutoML")
            model = AutoML(model_path) if os.path.exists(model_path) else None

        except Exception as e:
            return dbc.Alert(f"Error loading the explainer: {e}", color="danger"), True

    else:
        # No hay explainer: entrenar/cargar AutoML según exista carpeta previa
        model, X_test, y_test, reportML, trained, df, loaded = create_AutoML_model(contents, filename)
        try:
            X_test = df.loc[X_test.index].drop(columns=["Target"])
            y_test = df.loc[X_test.index]["Target"]

            X_test_fixed = X_test.copy()
            y_test_fixed = y_test.copy()
            X_test_fixed.index = X_test.index
            y_test_fixed.index = X_test.index

            explainer = ClassifierExplainer(
                trained,
                X_test_fixed,
                y_test_fixed,
                labels=["Dropout", "No dropout"],
                target="Target",
                #permutation_importance=False,
            )
            disable_permutation(explainer)

            explainer.X = X_test_fixed
            explainer.y = y_test_fixed
            explainer.index_backup = X_test_fixed.index.tolist()

            # Guardar X_test/y_test y explainer
            with open(os.path.join(saved_dir, "X_test.pkl"), "wb") as f:
                pickle.dump(X_test_fixed, f)
            with open(os.path.join(saved_dir, "y_test.pkl"), "wb") as f:
                pickle.dump(y_test_fixed, f)

            ensure_shap_inside_explainer(explainer, saved_explainer_path)
            print("Correctly saved the explainer with index:", explainer.X.index[:5])

        except Exception as e:
            return dbc.Alert(f"Error generating the explainer: {e}", color="danger"), True

    # --- Si llegamos aquí, tenemos explainer listo: construir tabs y lanzar dashboard

    group_tab = GroupCounterfactualsTab(explainer, name="group", title="Group Counterfactuals")
    group_tab.register_callbacks(app)

    try:
        counter_tab = CounterfactualsTab(explainer=explainer, dataframe=df, trained_model=trained)
    except Exception as e:
        print(f"❌ Error loading CounterfactualsTab: {e}")
        counter_tab = None

    if counter_tab:
        counter_tab.counterfactual.component_callbacks(app)

    tabs = []
    if reportML is not None:
        tabs.append(AutoMLReportTab(explainer=explainer, ML_report=reportML, title="AutoML report"))

    tabs.extend([
        FeaturesImportanceExpertTab(explainer, title="Feature Impact"),
        ClassificationStatsTab(explainer, title="Classification statistics"),
        WhatIfExpertTab(explainer, title="What happens if..."),
        group_tab
    ])

    group_tab.title = "Cluster Group Counterfactuals"

    if counter_tab:
        counter_tab.title = "Counterfactuals scenarios"
        tabs.append(counter_tab)


    db2 = ExplainerDashboard(
        explainer,
        header_hide_selector=True,
        hide_poweredby=True,
        title="Dashboards hub",
        tabs=tabs,
        theme='flatly',
        use_config=False,
        css=css_path
    )

    def start_dashboard(db, port):
        try:
            db.run(port=port)
            print(f"✅ Dashboard correctly started in http://localhost:{port}")
        except Exception as e:
            print(f"❌ Error starting the dashboard in port {port}: {e}")

    threading.Thread(target=lambda: start_dashboard(db2, 8055)).start()

    return None, html.Div([
        html.H1("Dropout Risk Monitoring Panel"),
        html.Iframe(
            src="http://localhost:8055",
            style={"width": "100%", "height": "1000px", "border": "none"}
        )
    ])





if __name__ == '__main__':
    app.title = "Dropout Insight: Educational Risk Dashboard with Counterfactual Explanations"
    webbrowser.open_new_tab('http://localhost:8050/')
    serve(app.server, host='0.0.0.0', port=8050)
