# Dropout Insight: Educational Risk Dashboard with Counterfactual Explanations

Herramienta interactiva para cargar datasets, entrenar automáticamente un modelo de *machine learning* o reutilizar uno ya entrenado, y visualizar e interpretar las predicciones del riesgo de abandono académico.

## 🚀 Características principales
- Carga de datasets en formato CSV.
- Entrenamiento automático de modelos de clasificación.
- Reutilización de modelos previamente generados.
- Visualización intuitiva de predicciones y factores de riesgo.
- Generación de explicaciones y contrafactuales (individuales y grupales).
- Dashboard interactivo desarrollado en **Plotly Dash**.


### 🛠️ Entorno virtual recomendado
Se recomienda crear un entorno virtual para instalar las dependencias de forma aislada.  

En sistemas basados en Python 3:

```bash
## Crear entorno virtual
python -m venv venv

## Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux / MacOS:
source venv/bin/activate

## 📦 Requisitos
Las dependencias se encuentran en `requirements.txt`.

Instalación con pip:

```bash
pip install -r requirements.txt
```


## ▶️ Iniciar la aplicación

Para ejecutar la app:

```bash
python src/index.py
```

La aplicación se abrirá automáticamente en [http://localhost:8050](http://localhost:8050).

## 🖥️ Uso

1. Seleccionar un dataset en formato **.csv**.  
2. El sistema comprueba si existe un modelo previamente guardado:
      - Si no existe, **se entrena automáticamente desde cero** con los datos cargados y se guarda dentro de una carpeta que si no existe se crea: **"/src/saved_models/nombre_archivo**
      - Si existe, **se reutiliza el modelo guardado en la carpeta correspondiente dentro de la carpeta correspondiente a ese archivo** para generara predicciones   
3. Una vez procesados los datos, navegar por las pestañas disponibles en el hub:  
   - **AutoML Report**
   - **Classification statistics**
   - **Features Importance**  
   - **What If Analysis**  
   - **Counterfactuals**  
   - **Group Counterfactuals**

## 📂 Estructura del repositorio

```
├── src/                # Código fuente de la aplicación
├── data/               # Datasets usados
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación principal
```

## 📖 Documentación adicional
- Manual de Usuario  


