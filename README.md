# Dropout Insight: Educational Risk Dashboard with Counterfactual Explanations

Interactive tool to upload datasets, automatically train a machine learning model or reuse one that is already trained, and visualize and interpret the risk of dropout in education.

## 🚀 Principal characteristics
- Upload files in .csv format
- Automatic training of classification models
- Reuse of previously generated models
- Intuitive visualization of predictions and risk factors
- Model explanations and counterfactuals generation
- Interactive dashboard developed in Plotly Dash

## 🛠️ Recommended virtual environment
Using a virtual environment is recommended to install dependencies in isolation.

Python 3 systems:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# Linux or macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Initialize the app

Run the application with:

```bash
python src/index.py
```

Access the dashboard at:
http://localhost:8050

## 🖥️ Use

1. Select a dataset in .csv format  
2. The system checks if a saved model exists:
   - If it does not exist, the model is trained automatically with the uploaded data. A folder is created if needed at: `src/saved_models/nombre_archivo`
   - If it exists, the saved model is loaded and used to generate predictions
3. Navigate through the tabs available in the dashboard:
   - AutoML Report
   - Classification statistics
   - Features importance
   - What If Analysis
   - Counterfactuals
   - Group Counterfactuals

## 📦 Running the project with Docker

This project includes a Dockerfile that allows running the app without installing Python or dependencies manually.

### 1. Build the Docker image

The Dockerfile is located inside the `src/` folder.  
From the root of the repository run:

```bash
docker build -t dropout-insight -f src/Dockerfile .
```

### 2. Run the container

Start the app:

```bash
docker run -p 8050:8050 dropout-insight
```

After running, open in your browser:

http://localhost:8050

## ⭐ Fast option to automatically open the browser

This repository includes optional helper scripts that start the container and automatically open the browser.

### Windows: `run.bat`

```bat
@echo off
start "" http://localhost:8050
docker run -p 8050:8050 dropout-insight
```

Usage: double click `run.bat`

### Linux or macOS: `run.sh`

```bash
#!/bin/bash
docker run -p 8050:8050 dropout-insight &
sleep 3
xdg-open "http://localhost:8050" 2>/dev/null || open "http://localhost:8050"
```

Usage:

```bash
chmod +x run.sh
./run.sh
```

### 3. Using local datasets inside Docker (optional)

```bash
docker run -p 8050:8050 -v "$(pwd)/data:/app/data" dropout-insight
```

Windows PowerShell:

```bash
docker run -p 8050:8050 -v ${PWD}\data:/app/data dropout-insight
```

### 4. Useful Docker commands

```bash
docker ps
docker stop <id>
docker rm <id>
docker rmi dropout-insight
```

## 📂 Repository structure

```
├── src/                # Application source code and Dockerfile
│   └── Dockerfile
├── data/               # Used datasets
├── doc/                # User manual
├── requirements.txt    # Dependencies
└── README.md           # Project summary
```

## 📖 Additional documentation
The complete user manual is located in: `doc/UserManual.pdf`
