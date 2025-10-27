# Dropout Insight: Educational Risk Dashboard with Counterfactual Explanations

Interactive tool to upload datasets, automatically train a machine learning model o reuse one that is already trained and visualize and interpretate the risk dropout predictions in education. 

## 🚀 Principal characteristics
- Upload the file in .csv format
- Automatical training of classification models
- Reuse of previous generated models
- Intuitive visualization of predictions and risk factors
- Model explications and counterfactuals generation
- Interactive Dashboard developed in **Plotly Dash**. 


### 🛠️ Recommended virtual enviroment 
It is strongly recommended to create a virtual enviroment to install the dependencies in isolation

In Python 3 based systems:

```bash
## Create the virtual enviroment
python -m venv venv

## Activate the virtual enviroment
# Windows:
venv\Scripts\activate
# Linux / MacOS:
source venv/bin/activate

## 📦 Requests
The dependencies can be found in 'requirements.txt'

Instalation with pip:

```bash
pip install -r requirements.txt
```


## ▶️ Initialize the app

For run the app:

```bash
python src/index.py
```

The App will open automatically in [http://localhost:8050](http://localhost:8050).

## 🖥️ Use

1. Select a dataset in **.csv** format.  
2. The sistem checks if there is and already saved model:
      - If it doesn't exist, **It is automatically trained from scratch** with the uploaded data and saved in a folder, that if it doesn't exist, it is created: **"/src/saved_models/nombre_archivo**
      - If it exist, **The model saved in the corresponding folder within the folder corresponding to that file is reused.** for generating predictions   
3. Once the data is processed, navigate throught the available tabs in the hub:  
   - **AutoML Report**
   - **Classification statistics**
   - **Features Importance**  
   - **What If Analysis**  
   - **Counterfactuals**  
   - **Group Counterfactuals**

## 📂 Repository structure

```
├── src/                # Source code for the application
├── data/               # Used Datasets
├── doc/                # User Manual
├── requirements.txt    # Project dependencies
└── README.md           # Project summary
```

## 📖 Aditional documentation
- User manual: The complete manual is found available in the folder: [`doc`](doc/UserManual.pdf).


