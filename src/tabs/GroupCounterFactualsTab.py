from dash import html, dcc, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from explainerdashboard.custom import ExplainerComponent

import numpy as np
import pandas as pd
import dice_ml as dml
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go


import plotly.graph_objects as go


#helper para ayudar a calcular las variables que han sido modificadas con el CF
def compute_changed_features(cf_df, original_1row_df, candidate_cols, tol=1e-9):
    if cf_df is None or cf_df.empty or "Error" in cf_df.columns:
        return []

    cols = pd.Index(candidate_cols).intersection(cf_df.columns).intersection(original_1row_df.columns)
    if len(cols) == 0:
        return []

    cf = cf_df.loc[cf_df.index[0], cols]
    orig = original_1row_df.loc[original_1row_df.index[0], cols]

    changed = []
    for col in cols:
        a, b = cf[col], orig[col]
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) != pd.isna(b):
            changed.append(col); continue
        try:
            fa, fb = float(a), float(b)
            if not np.isclose(fa, fb, atol=tol):
                changed.append(col)
        except Exception:
            if str(a) != str(b):
                changed.append(col)
    return changed


class GroupCounterfactualsTab(ExplainerComponent):
    def __init__(self, explainer, name=None, title="Group Counterfactuals", **kwargs):
        super().__init__(explainer, name=name, title=title, **kwargs)
        self.clustered_df = None

        self.explainer = explainer

    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H3("Group Counterfactuals", id='group-cf-title-'+str(self.name)),
                        html.H6("Choose clustering method and number of clusters to create groups of similar students.",
                                className='card-subtitle'),
                    ]),
                ]),
                dbc.CardBody([
                    # Controls of clustering 
                    dbc.Row([
                        dbc.Col([
                            html.Label("Clustering Mode:"),
                            dcc.RadioItems(
                                id='clustering-mode',
                                options=[
                                    {'label': 'Automatic (optimal k)', 'value': 'auto'},
                                    {'label': 'Manual (choose k)', 'value': 'manual'}
                                ],
                                value='auto',
                                labelStyle={'display': 'block'}
                            )
                        ], width=4, md=4),

                        dbc.Col([
                            html.Label("Number of Clusters (k):"),
                            dcc.Input(id='num-clusters', type='number', min=2, max=20, step=1, value=3)
                        ], width=3, md=3),

                        dbc.Col([
                            html.Br(),
                            dbc.Button("Create Clusters", id='cluster-button', color='primary')
                        ], width=3, md=3),
                    ], class_name="mb-2"),

                    html.Hr(),

                    # Clustering
                    dbc.Row([
                        dbc.Col([ html.Div(id='cluster-output') ], width=12)
                    ], class_name="mb-3"),

                    # Clúster selection
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select cluster to analyze:"),
                            dcc.Dropdown(id='selected-cluster')
                        ], width=8, md=8),
                    ], class_name="mb-3"),

                    # Results
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Generate Group Counterfactuals",
                                    id='generate-group-cf-button', color='success')
                        ], width=4, md=4),
                    ], class_name="mb-2"),

                    #html.Div(id='group-cf-output')

                    dbc.Progress(
                    id='cf-progress',
                    value=100,
                    striped=True,
                    animated=True,
                    color="info",
                    label="Generating group counterfactuals...",
                    style={"display": "none", "marginBottom": "12px"}
                    ),

                    # Output (opcional con ruedecita)
                    dcc.Loading(
                        id="cf-loading-overlay",
                        type="default",
                        children=html.Div(id='group-cf-output')
                    ),
                ]),
        ], class_name="mt-3 shadow-sm h-100"),

        dcc.Store(id='clustered-data-store'),

        dcc.Store(id='cf-loading-store', data=False),


        ], fluid = True)

    def register_callbacks(self, app):

        # Calback para crear clusteres
        @app.callback(
            Output('cluster-output', 'children'),
            Output('clustered-data-store', 'data'),
            Input('cluster-button', 'n_clicks'),
            State('clustering-mode', 'value'),
            State('num-clusters', 'value')
        )
        def cluster_students(n_clicks, mode, k):
            if not n_clicks:
                raise PreventUpdate

            import pandas as pd
            from dash import html, dcc, dash_table
            import plotly.express as px
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            # 1) Datos base para clústeres (solo features)
            df = self.explainer.X.copy()

            # Por si 'cluster' quedó de una ejecución previa
            if 'cluster' in df.columns:
                df = df.drop(columns=['cluster'])

            # 2) Solo columnas numéricas para KMeans
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(num_cols) == 0:
                msg = html.Div([
                    html.H5("Cannot create clusters"),
                    html.P("No numeric features available for clustering.")
                ], style={"border": "1px solid #dc3545", "padding": "10px", "borderRadius": "6px", "background": "#fff5f5"})
                return msg, None

            # 3) Si hay NaNs en numéricas → informar en la pestaña y NO ejecutar KMeans
            nan_counts = df[num_cols].isna().sum()
            total_nans = int(nan_counts.sum())
            if total_nans > 0:
                nan_table = dash_table.DataTable(
                    columns=[{"name": "Feature", "id": "feature"},
                            {"name": "NaN count", "id": "nans"}],
                    data=[{"feature": c, "nans": int(n)}
                        for c, n in nan_counts[nan_counts > 0].sort_values(ascending=False).items()],
                    style_cell={"textAlign": "left", "padding": "6px"},
                    style_header={"fontWeight": "bold"},
                    page_size=10
                )
                msg = html.Div([
                    html.H5("Cannot create clusters"),
                    html.P("There are missing values (NaN) in the numeric features. Please clean/impute them and try again."),
                    nan_table
                ], style={"border": "1px solid #dc3545", "padding": "10px", "borderRadius": "6px", "background": "#fff5f5"})
                return msg, None  # mostramos el error y dejamos vacío el Store

            # 4) Chequeos extra para evitar errores comunes
            # 4.1 Suficientes filas para k
            if df.shape[0] < max(2, k):
                msg = html.Div([
                    html.H5("Cannot create clusters"),
                    html.P(f"Not enough rows for k={k}. Please increase data size or reduce k.")
                ], style={"border": "1px solid #dc3545", "padding": "10px", "borderRadius": "6px", "background": "#fff5f5"})
                return msg, None

            # 4.2 Variancia nula (todas constantes) en numéricas
            if (df[num_cols].std(numeric_only=True) == 0).all():
                msg = html.Div([
                    html.H5("Cannot create clusters"),
                    html.P("Numeric features are constant. Clustering is not meaningful.")
                ], style={"border": "1px solid #dc3545", "padding": "10px", "borderRadius": "6px", "background": "#fff5f5"})
                return msg, None

            # 5) Elegir k (auto o manual) sobre numéricas
            if mode == 'auto':
                best_score, best_k = -1, 2
                k_max = max(2, min(10, df.shape[0]-1))  # no probar k >= n_rows
                for i in range(2, k_max + 1):
                    try:
                        km = KMeans(n_clusters=i, random_state=42, n_init=10).fit(df[num_cols])
                        score = silhouette_score(df[num_cols], km.labels_)
                        if score > best_score:
                            best_k, best_score = i, score
                    except Exception:
                        continue
                k = best_k

            # 6) Ajustar KMeans (ya sin NaNs y solo numéricas)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(df[num_cols])
            df['cluster'] = kmeans.labels_
            self.clustered_df = df  # guarda en memoria

            # 7) Salida visual + persistencia en Store
            cluster_counts = df['cluster'].value_counts().sort_index().reset_index()
            cluster_counts.columns = ['Cluster', 'Num Students']
            fig = px.bar(cluster_counts, x='Cluster', y='Num Students',
                        title=f"Number of Students per Cluster (k={k})")

            content = html.Div([
                html.H5(f"Created {k} clusters."),
                dcc.Graph(figure=fig)
            ])

            return content, df.to_json(date_format='iso', orient='split')


        # Callback to poblate the dropwdown with clusteres
        @app.callback(
            Output('selected-cluster', 'options'),
            Input('clustered-data-store', 'data')
        )
        def update_cluster_dropdown(clustered_json):
            if not clustered_json:
                raise PreventUpdate
            df = pd.read_json(clustered_json, orient='split')
            self.clustered_df = df  # ← rehidrata memoria por si se perdió
            clusters = sorted(df['cluster'].unique())
            return [{"label": f"Cluster {i}", "value": i} for i in clusters]

        # Callback fpr the counterfactuals creation

        @app.callback(
            Output('group-cf-output', 'children'),
            Input('generate-group-cf-button', 'n_clicks'),
            State('selected-cluster', 'value'),
            State('clustered-data-store', 'data'),
            prevent_initial_call=True
        )
        def generate_group_counterfactuals(n_clicks, selected_cluster,clustered_json):
            # 0) Guardas iniciales de interacción
            if not n_clicks:
                raise PreventUpdate

            if self.clustered_df is None:
                return html.Div("⚠️ No cluster data found. Click 'Create Clusters' first and wait for the bar chart.")

            df = self.clustered_df.copy()

            if 'cluster' not in df.columns:
                return html.Div("⚠️ No cluster labels found. Please click 'Create Clusters' first.")

            if selected_cluster is None:
                return html.Div("⚠️ Select a cluster from the dropdown first.")

            # 1) Preparar df_full para DiCE (con target)
            df_full = self.explainer.X.copy()
            df_full["Target"] = self.explainer.y
            outcome_col = "Target"

            if 'cluster' in df_full.columns:
                df_full = df_full.drop(columns=['cluster'])


            # Limpieza de columnas
            df_full_clean = df_full.loc[:, df_full.columns.notnull()]

            # Identificar continuas/categóricas
            continuous_all = df_full_clean.select_dtypes(include=["float64", "int64"]).columns.tolist()
            continuous = [col for col in continuous_all if col != outcome_col and col in df_full_clean.columns]
            categorical = [col for col in df_full_clean.columns if col not in continuous and col != outcome_col]

            # Validaciones
            assert all(col in df_full_clean.columns for col in continuous), "❌ Invalid continuous features "
            assert outcome_col in df_full_clean.columns, "❌ Target not founded in df_full_clean"

            # 2) Construir objeto Data de DiCE
            data = dml.Data(
                dataframe=df_full_clean,
                continuous_features=continuous,
                categorical_features=categorical,
                outcome_name=outcome_col
            )

            # 3) Envolver el modelo para DiCE
            from dice_ml.model_interfaces.base_model import BaseModel

            class CustomModel(BaseModel):
                def __init__(self, model):
                    super().__init__(model=model, backend="sklearn")
                    self.model_type = "classifier"

                def predict(self, input_data):
                    return self.model.predict(input_data)

                def predict_proba(self, input_data):
                    return self.model.predict_proba(input_data)

            class WrappedModel:
                def __init__(self, clf):
                    self.model = clf
                    self.classes_ = [0, 1]

                def predict(self, X):
                    return self.model.predict(X)

                def predict_proba(self, X):
                    return self.model.predict_proba(X)

            model_final = self.explainer.model
            if hasattr(model_final, "model"):
                model_final = model_final.model

            model = CustomModel(WrappedModel(model_final))
            exp = dml.Dice(data, model, method="random")

            # 4) Seleccionar cluster y representante
            results = []
            modified_cluster_df = None

            cluster_id = selected_cluster
            cluster_df = df.loc[df['cluster'] == cluster_id].copy()
            if cluster_df.empty:
                return html.Div(f"⚠️ Cluster {cluster_id} has no rows.")

            if 'cluster' in cluster_df.columns:
                cluster_df.drop(columns=['cluster'], inplace=True)

            representative = cluster_df.sample(n=1, random_state=42)

            # 5) Generar CF
            try:
                dice_exp = exp.generate_counterfactuals(representative, total_CFs=1, desired_class="opposite")
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                cf_df[outcome_col] = cf_df[outcome_col].replace({0: "Dropout", 1: "No dropout"})
            except Exception as e:
                print(f"❌ Error generating counterfactuals: {e}")
                cf_df = pd.DataFrame({"Error": [str(e)]})

            # 6) Tabla CF + original
            changed_features = compute_changed_features(
                cf_df=cf_df,
                original_1row_df=representative,
                candidate_cols=cluster_df.columns
            )

            green_highlight = [
                {"if": {"column_id": col},
                "backgroundColor": "#e6f4ea",
                "color": "#1a7f37",
                "fontWeight": "700"} for col in changed_features if col in cf_df.columns
            ]

            table = dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in cf_df.columns],
                data=cf_df.to_dict("records"),
                style_table={"overflowX": "scroll"},
                style_cell={"textAlign": "left", "padding": "6px"},
                style_header={"fontWeight": "bold"},
                style_data_conditional=green_highlight,
                page_size=5
            )

            results.append(html.Div([
                html.H5(f"Cluster {cluster_id} - Counterfactuals"),
                table,
                html.Hr()
            ]))

            original_table = dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in representative.columns],
                data=representative.to_dict("records"),
                style_table={"overflowX": "scroll"},
                style_cell={"textAlign": "left", "padding": "6px"},
                style_header={"fontWeight": "bold"},
                page_size=5
            )

            results.append(html.Div([
                html.H5(f"Cluster {cluster_id} - Original student (representative)"),
                original_table,
                html.Hr()
            ]))

            # 7) Comparación antes/después en el clúster
            if not cf_df.empty and outcome_col in cf_df.columns:
                cf_row = cf_df.iloc[0]
                original_row = representative.iloc[0]

                changed_features = [
                    col for col in cf_row.index
                    if col in cluster_df.columns and cf_row[col] != original_row[col]
                ]

                if not changed_features:
                    results.append(html.Div("⚠️ There are no modified features to apply in the cluster."))
                    return results

                modified_cluster_df = cluster_df.copy()
                for feature in changed_features:
                    modified_cluster_df[feature] = cf_row[feature]

                prob_original = self.explainer.model.predict_proba(cluster_df)[:, 1]
                preds_original = (prob_original >= 0.5).astype(int)

                prob_modified = self.explainer.model.predict_proba(modified_cluster_df)[:, 1]
                preds_modified = (prob_modified >= 0.5).astype(int)

                counts_original = pd.Series(preds_original).value_counts().sort_index()
                counts_modified = pd.Series(preds_modified).value_counts().sort_index()

                fig = go.Figure(data=[
                    go.Bar(name='Original', x=['No dropout', 'Dropout'],
                        y=[counts_original.get(1, 0), counts_original.get(0, 0)]),
                    go.Bar(name='Counterfactual applied', x=['No dropout', 'Dropout'],
                        y=[counts_modified.get(1, 0), counts_modified.get(0, 0)])
                ])
                fig.update_layout(
                    barmode='group',
                    title='Cluster prediction distribution (Before vs after applying the counterfactual)',
                    yaxis_title='Count of students'
                )
                results.append(dcc.Graph(figure=fig))
            else:
                results.append(html.Div("⚠️ No counterfactuals to apply (empty CF result)."))
                return results

            # 8) Si no se creó modified_cluster_df, paramos aquí
            if modified_cluster_df is None:
                return results

            # 9) Histograma cambios por alumno
            num_changes_list = []
            for idx in cluster_df.index:
                original_row = cluster_df.loc[[idx]]
                modified_row = modified_cluster_df.loc[[idx]]
                num_changes = (original_row.values != modified_row.values).sum()
                num_changes_list.append(num_changes)

            nbins_value = int(max(num_changes_list)) + 1
            hist_fig = px.histogram(
                x=num_changes_list,
                nbins=nbins_value,
                labels={'x': 'Number of modified features', 'y': 'Nº of students'},
                title='Distribution of the number of modified features when applying the counterfactual'
            )
            hist_fig.update_layout(bargap=0.2, height=500)
            results.append(dcc.Graph(figure=hist_fig))

            # 10) Boxplot del “gain” de probabilidad
            gain_list = []
            for idx in cluster_df.index:
                orig_row = cluster_df.loc[[idx]]
                mod_row = modified_cluster_df.loc[[idx]]
                prob_orig = self.explainer.model.predict_proba(orig_row)[0][0]
                prob_cf = self.explainer.model.predict_proba(mod_row)[0][0]
                gain = prob_cf - prob_orig
                gain_list.append(gain)

            gain_df = pd.DataFrame({
                "Cluster": [f"Clúster {selected_cluster}"] * len(gain_list),
                "Probability gain": gain_list
            })
            box_fig = px.box(
                gain_df, x="Cluster", y="Probability gain", points="all",
                title="Distribution of probability gain after applying counterfactual"
            )
            box_fig.update_layout(yaxis_title="P(CF) - P(Original)", height=400)
            results.append(dcc.Graph(figure=box_fig))

            # 11) Tabla resumen
            num_students = len(cluster_df)
            preds_orig = self.explainer.model.predict(cluster_df)
            preds_cf = self.explainer.model.predict(modified_cluster_df)
            pct_dropout_orig = 100 * (preds_orig == 0).sum() / num_students
            pct_dropout_cf = 100 * (preds_cf == 0).sum() / num_students

            num_changes_list = [
                (cluster_df.loc[[idx]].values != modified_cluster_df.loc[[idx]].values).sum()
                for idx in cluster_df.index
            ]
            avg_changes = round(np.mean(num_changes_list), 2)
            mod_counts = {col: (cluster_df[col] != modified_cluster_df[col]).sum() for col in cluster_df.columns}
            top_vars = sorted(mod_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_vars_names = ', '.join([v[0] for v in top_vars if v[1] > 0]) or "Ninguna"

            summary_data = [
                {"Metric": "Number of students in the cluster", "Value": num_students},
                {"Metric": "% Original Dropout", "Value": f"{pct_dropout_orig:.1f}%"},
                {"Metric": "% Dropout after counterfactual", "Value": f"{pct_dropout_cf:.1f}%"},
                {"Metric": "Average number of modified variables", "Value": avg_changes},
                {"Metric": "Top most modified variables", "Value": top_vars_names},
            ]
            summary_table = dash_table.DataTable(
                columns=[{"Name": i, "id": i} for i in ["Metric", "Value"]],
                data=summary_data,
                style_cell={"textAlign": "left", "padding": "6px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f2f2f2"},
                style_table={"marginTop": "20px", "width": "80%"}
            )
            results.append(html.Br())
            results.append(html.H5("Summary of the cluster after applying the counterfactuals"))
            results.append(summary_table)

            return results

