# ForestIQ — Forest Fire Risk Prediction

## 📌 Project Overview

**ForestIQ** is a machine-learning-based forest fire risk prediction system that predicts whether forest conditions are likely to result in a **fire or no-fire condition** based on environmental and fire-weather parameters.

The project uses the **Algerian Forest Fires dataset** and applies supervised machine-learning techniques including **Logistic Regression, Decision Tree, Random Forest, and XGBoost**.

The trained models are deployed using **FastAPI** and connected to an interactive web dashboard for real-time prediction and monitoring.

---

## 🎯 Objectives

* Analyze forest-fire environmental data.
* Clean and preprocess the raw dataset.
* Select relevant features for prediction.
* Train multiple machine-learning classification models.
* Compare model performance.
* Identify important fire-risk features.
* Deploy the trained model through a REST API.
* Provide an interactive **ForestIQ** dashboard.
* Monitor prediction probability, risk level, and API performance.

---

## 📊 Dataset

The project uses the **Algerian Forest Fires Dataset**.

### Original dataset

* **Records:** 246
* **Attributes:** 14

| Attribute     | Description                    |
| ------------- | ------------------------------ |
| `day`         | Day of observation             |
| `month`       | Month of observation           |
| `year`        | Year of observation            |
| `Temperature` | Air temperature in °C          |
| `RH`          | Relative humidity (%)          |
| `Ws`          | Wind speed                     |
| `Rain`        | Rainfall                       |
| `FFMC`        | Fine Fuel Moisture Code        |
| `DMC`         | Duff Moisture Code             |
| `DC`          | Drought Code                   |
| `ISI`         | Initial Spread Index           |
| `BUI`         | Buildup Index                  |
| `FWI`         | Fire Weather Index             |
| `Classes`     | Fire / Not Fire classification |

### Selected features

The ForestIQ prediction system uses **9 input features**:

```text
Temperature
RH
Ws
Rain
FFMC
DMC
DC
ISI
BUI
```

The `Classes` column is converted into the target:

```text
Fire     → 1
Not Fire → 0
```

After preprocessing:

* **243 records**
* **9 input features**
* **1 prediction target**

---

## 🧠 Machine Learning Techniques

### Supervised Learning

The primary approach is **supervised classification** because the dataset contains known Fire/Not Fire labels.

The project implements:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **XGBoost**

### Unsupervised Analysis

The project also contains an `unsupervised.py` module for additional exploratory analysis using techniques such as:

* K-Means
* Hierarchical Clustering
* DBSCAN
* PCA
* t-SNE / UMAP
* Anomaly detection

These techniques are supplementary; the main ForestIQ prediction is supervised learning.

---

## 📈 Model Performance

| Model         | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------- | -------: | --------: | -----: | -------: | ------: |
| Decision Tree |   93.88% |    90.32% |   100% |   94.92% |  92.86% |
| Random Forest |     100% |      100% |   100% |     100% |    100% |
| XGBoost       |   95.92% |    93.33% |   100% |   96.55% |    100% |

> These scores are based on the project's evaluated train/test split.

---

## 🔍 Feature Importance

Random Forest feature importance showed the following ranking:

| Feature     | Importance |
| ----------- | ---------: |
| ISI         |     39.90% |
| FFMC        |     31.13% |
| DC          |     11.11% |
| BUI         |      6.49% |
| DMC         |      5.51% |
| Rain        |      3.94% |
| Temperature |      1.05% |
| RH          |      0.67% |
| Ws          |      0.21% |

**ISI** was the most important feature in the trained Random Forest model.

---

# 🏗️ Project Architecture

```text
                    Algerian Forest Fire Dataset
                               │
                               ▼
                       Data Inspection
                               │
                               ▼
                       Data Preprocessing
                               │
                               ▼
                        Feature Selection
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Machine Learning    │
                    │                     │
                    │ Logistic Regression │
                    │ Decision Tree       │
                    │ Random Forest       │
                    │ XGBoost             │
                    └──────────┬──────────┘
                               │
                               ▼
                       Model Evaluation
                               │
                               ▼
                         Model Saving
                               │
                               ▼
                         FastAPI Backend
                               │
                               ▼
                       ForestIQ Dashboard
                               │
                               ▼
                  Fire / Not Fire + Risk Level
```

---

# 📁 Project Structure

```text
Forest-Fire-Risk-Prediction/
│
├── data/
│   ├── Algerian_forest_fires_dataset_UPDATE.csv
│   └── forest_fire_processed.csv
│
├── src/
│   ├── inspect_data.py
│   ├── preprocess.py
│   ├── logistic_regression.py
│   ├── tree_models.py
│   ├── evaluation.py
│   ├── unsupervised.py
│   └── save_model.py
│
├── models/
│   ├── features.pkl
│   ├── forest_fire_random_forest.pkl
│   └── forest_fire_xgboost.pkl
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── index.html
│
├── results/
│   ├── tree_model_comparison.csv
│   ├── random_forest_feature_importance.csv
│   └── prediction_logs.csv
│
├── venv/
│
└── README.md
```

---

# 🧩 Module Description

### `inspect_data.py`

Inspects the raw dataset and displays:

* Dataset dimensions
* Column names
* Data types
* Sample records
* Dataset information

### `preprocess.py`

Performs:

* Data cleaning
* Column normalization
* Invalid-record handling
* Target conversion
* Feature selection
* Processed dataset generation

### `logistic_regression.py`

Implements Logistic Regression as a baseline classification model.

### `tree_models.py`

Implements:

* Decision Tree
* Random Forest
* XGBoost

and compares their performance.

### `evaluation.py`

Performs model evaluation using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Other evaluation analysis

### `unsupervised.py`

Performs additional exploratory unsupervised machine-learning analysis.

### `save_model.py`

Saves trained models and feature information using **Joblib**.

### `api/main.py`

Creates the FastAPI backend.

Main functionality includes:

```text
/predict
/health
/model-info
/monitoring
```

The `/predict` endpoint receives the nine dashboard inputs and returns the prediction.

### `dashboard/index.html`

Provides the **ForestIQ** user interface.

The dashboard allows users to enter:

```text
Temperature
RH
Ws
Rain
FFMC
DMC
DC
ISI
BUI
```

and view:

* Fire/Not Fire prediction
* Fire probability
* Risk level
* Model information
* Prediction monitoring
* API latency

---

# 🔄 Prediction Flow

When a user enters environmental values:

```text
User Input
   ↓
ForestIQ Dashboard
   ↓
POST /predict
   ↓
FastAPI
   ↓
Feature Validation
   ↓
XGBoost / Random Forest
   ↓
Fire Probability
   ↓
Risk Classification
   ↓
Dashboard Result
```

For example:

```text
Fire Probability: 99.31%
Prediction: Fire
Risk Level: VERY HIGH
```

---

# 🚦 Risk Classification

ForestIQ converts the predicted Fire probability into a risk category:

|  Probability | Risk      |
| -----------: | --------- |
|      `< 30%` | LOW       |
| `30% – <60%` | MEDIUM    |
| `60% – <80%` | HIGH      |
|      `≥ 80%` | VERY HIGH |

---

# 🌐 Running the Project

## 1. Activate the virtual environment

Open PowerShell:

```powershell
cd "C:\Users\jayan\Desktop\Forest-Fire-Risk-Prediction"
```

Activate the environment:

```powershell
.\venv\Scripts\activate
```

You should see:

```text
(venv) PS C:\Users\jayan\Desktop\Forest-Fire-Risk-Prediction>
```

---

## 2. Start the FastAPI backend

Because `main.py` is inside the `api` folder:

```powershell
python -m uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

---

## 3. Test the API

Open:

```text
http://127.0.0.1:8000/docs
```

This opens the FastAPI Swagger interface where the API endpoints can be tested.

Health check:

```text
http://127.0.0.1:8000/health
```

---

## 4. Open the ForestIQ dashboard

Open another PowerShell window:

```powershell
cd "C:\Users\jayan\Desktop\Forest-Fire-Risk-Prediction"
```

Then:

```powershell
start ".\dashboard\index.html"
```

Or open manually:

```text
Forest-Fire-Risk-Prediction
└── dashboard
    └── index.html
```

**Keep the FastAPI terminal running while using the dashboard.**

---

# 🛠️ Technologies Used

### Programming

* Python
* HTML
* CSS
* JavaScript

### Machine Learning

* Scikit-learn
* XGBoost

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

### Deployment

* FastAPI
* Uvicorn
* Pydantic
* CORS

### Model Persistence

* Joblib

---

# 🎓 Course Outcome Mapping

| CO      | Project Implementation                         |
| ------- | ---------------------------------------------- |
| **CO1** | Dataset inspection and preprocessing           |
| **CO2** | Feature selection and classification           |
| **CO3** | Decision Tree, Random Forest and XGBoost       |
| **CO4** | Model evaluation and feature importance        |
| **CO5** | Model saving and FastAPI deployment            |
| **CO6** | ForestIQ dashboard and application integration |

---

# 🔬 Key Techniques

The project demonstrates:

* Data preprocessing
* Feature selection
* Binary classification
* Logistic Regression
* Decision Trees
* Ensemble learning
* Random Forest
* Gradient boosting
* XGBoost
* Model evaluation
* Feature importance analysis
* Unsupervised learning
* REST API development
* Model deployment
* Real-time prediction
* Prediction logging
* Dashboard visualization

---

# 👨‍💻 Project Summary

**ForestIQ** combines machine learning and web deployment to create an interactive forest-fire prediction system.

The system takes **nine environmental/fire-weather attributes**, sends them to a trained machine-learning model through a **FastAPI REST API**, and returns a **Fire/Not Fire prediction, probability, and risk level**.

The project demonstrates the complete machine-learning pipeline:

```text
Data
 → Cleaning
 → Feature Selection
 → Model Training
 → Evaluation
 → Model Saving
 → API Deployment
 → Web Dashboard
 → Prediction Monitoring
```

**Project:** ForestIQ — Forest Fire Risk Prediction
**Primary ML approach:** Supervised Classification
**Main Models:** Random Forest & XGBoost
**Backend:** FastAPI
**Frontend:** HTML/CSS/JavaScript
**Dataset:** Algerian Forest Fires Dataset
