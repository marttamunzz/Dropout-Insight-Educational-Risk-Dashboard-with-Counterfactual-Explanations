# Dropout Insight: Educational Risk Dashboard with Counterfactual Explanations

Herramienta interactiva para cargar datasets, entrenar automáticamente un modelo de *machine learning* o reutilizar uno ya entrenado, y visualizar e interpretar las predicciones del riesgo de abandono académico.

## 🚀 Características principales
- Carga de datasets en formato CSV.
- Entrenamiento automático de modelos de clasificación.
- Reutilización de modelos previamente generados.
- Visualización intuitiva de predicciones y factores de riesgo.
- Generación de explicaciones y contrafactuales (individuales y grupales).
- Dashboard interactivo desarrollado en **Plotly Dash**.

## 📦 Requisitos
Las dependencias se encuentran en `requirements.txt`.

Instalación con pip:

```bash
pip install -r requirements.txt
```

## ▶️ Iniciar la aplicación

Para ejecutar la app:

```bash
python src/app.py
```

La aplicación se abrirá automáticamente en [http://localhost:8050](http://localhost:8050).

## 🖥️ Uso

1. Seleccionar un dataset en formato **.csv**.  
2. Elegir si se quiere **entrenar un nuevo modelo** o **cargar uno existente**.  
3. Explorar los resultados en las distintas pestañas:  
   - **Features Importance**  
   - **What If Analysis**  
   - **Counterfactuals**  

## 📂 Estructura del repositorio

```
├── src/                # Código fuente de la aplicación
├── data/               # Datasets (no se suben al repo)
├── readme_images/      # Imágenes y gifs para el README
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación principal
```

## 📖 Documentación adicional
- Manual de Usuario  


