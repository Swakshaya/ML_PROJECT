from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import joblib
import time
import os
import csv

from datetime import datetime


# ============================================================
# FORESTIQ - FOREST FIRE RISK INTELLIGENCE API
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

RF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "forest_fire_random_forest.pkl"
)

XGB_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "forest_fire_xgboost.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "features.pkl"
)

LOG_PATH = os.path.join(
    RESULTS_DIR,
    "prediction_logs.csv"
)


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(RF_MODEL_PATH):

    raise FileNotFoundError(
        f"""
Random Forest model not found:

{RF_MODEL_PATH}

Run:

python src/tree_models.py
"""
    )


if not os.path.exists(XGB_MODEL_PATH):

    raise FileNotFoundError(
        f"""
XGBoost model not found:

{XGB_MODEL_PATH}

Run:

python src/tree_models.py
"""
    )


if not os.path.exists(FEATURES_PATH):

    raise FileNotFoundError(
        f"""
Features file not found:

{FEATURES_PATH}

Run your model saving script first.
"""
    )


# ============================================================
# 4. LOAD MODELS
# ============================================================

random_forest = joblib.load(
    RF_MODEL_PATH
)

xgboost_model = joblib.load(
    XGB_MODEL_PATH
)

features = joblib.load(
    FEATURES_PATH
)


print()
print("=" * 70)
print("FORESTIQ AI ENGINE")
print("=" * 70)

print("Random Forest : LOADED")
print("XGBoost       : LOADED")
print("Features      :", features)
print("=" * 70)
print()


# ============================================================
# 5. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="ForestIQ",

    description=(
        "AI-powered wildfire risk intelligence "
        "using Random Forest and XGBoost."
    ),

    version="2.0.0"
)


# ============================================================
# 6. CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# 7. INPUT MODEL
# ============================================================

class FirePredictionRequest(BaseModel):

    Temperature: float

    RH: float

    Ws: float

    Rain: float

    FFMC: float

    DMC: float

    DC: float

    ISI: float

    BUI: float


# ============================================================
# 8. ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "application": "ForestIQ",

        "description":
            "AI-Powered Wildfire Risk Intelligence",

        "models": [
            "Random Forest",
            "XGBoost"
        ],

        "status": "online",

        "documentation":
            "/docs"
    }


# ============================================================
# 9. HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "application": "ForestIQ",

        "random_forest":
            "loaded",

        "xgboost":
            "loaded",

        "number_of_features":
            len(features)
    }


# ============================================================
# 10. MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "application":
            "ForestIQ",

        "models": {

            "random_forest":
                "Random Forest",

            "xgboost":
                "XGBoost"
        },

        "target":
            "Fire",

        "classes": {

            "0":
                "Not Fire",

            "1":
                "Fire"
        },

        "number_of_features":
            len(features),

        "features":
            features
    }


# ============================================================
# 11. RISK CLASSIFICATION
# ============================================================

def calculate_risk(probability):

    if probability < 0.30:

        return "LOW"

    elif probability < 0.60:

        return "MEDIUM"

    elif probability < 0.80:

        return "HIGH"

    else:

        return "VERY HIGH"


# ============================================================
# 12. PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    request: FirePredictionRequest
):

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    input_data = {

        "Temperature":
            request.Temperature,

        "RH":
            request.RH,

        "Ws":
            request.Ws,

        "Rain":
            request.Rain,

        "FFMC":
            request.FFMC,

        "DMC":
            request.DMC,

        "DC":
            request.DC,

        "ISI":
            request.ISI,

        "BUI":
            request.BUI
    }


    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [input_data]
    )

    input_df = input_df[
        features
    ]


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_prediction = int(
        random_forest.predict(
            input_df
        )[0]
    )

    rf_probability = float(
        random_forest.predict_proba(
            input_df
        )[0][1]
    )


    # ========================================================
    # XGBOOST
    # ========================================================

    xgb_prediction = int(
        xgboost_model.predict(
            input_df
        )[0]
    )

    xgb_probability = float(
        xgboost_model.predict_proba(
            input_df
        )[0][1]
    )


    # ========================================================
    # ENSEMBLE
    # ========================================================

    probability = (
        rf_probability +
        xgb_probability
    ) / 2


    # Both models must agree
    model_agreement = (
        rf_prediction ==
        xgb_prediction
    )


    # Final prediction
    prediction = (
        1
        if probability >= 0.50
        else 0
    )


    # Result
    result = (
        "Fire"
        if prediction == 1
        else "Not Fire"
    )


    # Risk
    risk_level = calculate_risk(
        probability
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = (
        probability
        if prediction == 1
        else 1 - probability
    )


    # ========================================================
    # LATENCY
    # ========================================================

    latency_ms = round(

        (
            time.perf_counter()
            - start_time
        ) * 1000,

        2
    )


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = datetime.now().isoformat()


    # ========================================================
    # MONITORING LOG
    # ========================================================

    log_data = {

        "timestamp":
            timestamp,

        "Temperature":
            request.Temperature,

        "RH":
            request.RH,

        "Ws":
            request.Ws,

        "Rain":
            request.Rain,

        "FFMC":
            request.FFMC,

        "DMC":
            request.DMC,

        "DC":
            request.DC,

        "ISI":
            request.ISI,

        "BUI":
            request.BUI,

        "prediction":
            prediction,

        "result":
            result,

        "fire_probability":
            probability,

        "risk_level":
            risk_level,

        "confidence":
            confidence,

        "random_forest_prediction":
            rf_prediction,

        "random_forest_probability":
            rf_probability,

        "xgboost_prediction":
            xgb_prediction,

        "xgboost_probability":
            xgb_probability,

        "model_agreement":
            model_agreement,

        "latency_ms":
            latency_ms
    }


    # ========================================================
    # WRITE LOG SAFELY
    # ========================================================

    log_df = pd.DataFrame(
        [log_data]
    )


    # If file doesn't exist, create it
    if not os.path.exists(LOG_PATH):

        log_df.to_csv(
            LOG_PATH,
            index=False
        )

    else:

        # Append with exactly the same columns
        log_df.to_csv(

            LOG_PATH,

            mode="a",

            header=False,

            index=False
        )


    # ========================================================
    # API RESPONSE
    # ========================================================

    return {

        "application":
            "ForestIQ",

        "prediction":
            prediction,

        "result":
            result,

        "risk_level":
            risk_level,

        "fire_probability":
            round(
                probability,
                4
            ),

        "fire_probability_percent":
            round(
                probability * 100,
                2
            ),

        "confidence_percent":
            round(
                confidence * 100,
                2
            ),

        "models": {

            "random_forest": {

                "prediction":
                    rf_prediction,

                "fire_probability_percent":
                    round(
                        rf_probability * 100,
                        2
                    )
            },

            "xgboost": {

                "prediction":
                    xgb_prediction,

                "fire_probability_percent":
                    round(
                        xgb_probability * 100,
                        2
                    )
            }
        },

        "model_agreement":
            model_agreement,

        "latency_ms":
            latency_ms,

        "timestamp":
            timestamp
    }


# ============================================================
# 13. LIVE MONITORING
# ============================================================

@app.get("/monitoring")
def monitoring():

    # --------------------------------------------------------
    # No log yet
    # --------------------------------------------------------

    if not os.path.exists(LOG_PATH):

        return {

            "status":
                "waiting",

            "total_predictions":
                0,

            "fire_predictions":
                0,

            "not_fire_predictions":
                0,

            "average_fire_probability":
                0,

            "average_latency_ms":
                0,

            "model_agreement_percent":
                0,

            "latest":
                None
        }


    # --------------------------------------------------------
    # Read log safely
    # --------------------------------------------------------

    try:

        logs = pd.read_csv(
            LOG_PATH
        )

    except Exception as e:

        return {

            "status":
                "waiting",

            "error":
                str(e),

            "total_predictions":
                0,

            "fire_predictions":
                0,

            "not_fire_predictions":
                0,

            "latest":
                None
        }


    # --------------------------------------------------------
    # Empty log
    # --------------------------------------------------------

    if logs.empty:

        return {

            "status":
                "waiting",

            "total_predictions":
                0,

            "fire_predictions":
                0,

            "not_fire_predictions":
                0,

            "average_fire_probability":
                0,

            "average_latency_ms":
                0,

            "model_agreement_percent":
                0,

            "latest":
                None
        }


    # ========================================================
    # TOTALS
    # ========================================================

    total_predictions = len(
        logs
    )


    fire_predictions = int(

        (
            pd.to_numeric(
                logs["prediction"],
                errors="coerce"
            )
            == 1
        ).sum()
    )


    not_fire_predictions = int(

        (
            pd.to_numeric(
                logs["prediction"],
                errors="coerce"
            )
            == 0
        ).sum()
    )


    # ========================================================
    # AVERAGE PROBABILITY
    # ========================================================

    probabilities = pd.to_numeric(

        logs["fire_probability"],

        errors="coerce"
    )


    average_probability = (

        float(
            probabilities.mean()
        )

        if not probabilities.empty

        else 0
    )


    # ========================================================
    # AVERAGE LATENCY
    # ========================================================

    latencies = pd.to_numeric(

        logs["latency_ms"],

        errors="coerce"
    )


    average_latency = (

        float(
            latencies.mean()
        )

        if not latencies.empty

        else 0
    )


    # ========================================================
    # MODEL AGREEMENT
    # ========================================================

    agreements = (

        logs["model_agreement"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
    )


    agreement_percent = (

        float(
            agreements.mean()
            * 100
        )

        if len(agreements) > 0

        else 0
    )


    # ========================================================
    # LATEST PREDICTION
    # ========================================================

    latest = logs.iloc[-1]


    def value(
        column,
        default=None
    ):

        if column not in latest:

            return default

        result = latest[column]

        if pd.isna(result):

            return default

        return result


    latest_probability = float(

        pd.to_numeric(

            value(
                "fire_probability",
                0
            ),

            errors="coerce"
        )
    )


    latest_prediction = int(

        pd.to_numeric(

            value(
                "prediction",
                0
            ),

            errors="coerce"
        )
    )


    latest_rf_probability = float(

        pd.to_numeric(

            value(
                "random_forest_probability",
                0
            ),

            errors="coerce"
        )
    )


    latest_xgb_probability = float(

        pd.to_numeric(

            value(
                "xgboost_probability",
                0
            ),

            errors="coerce"
        )
    )


    latest_rf_prediction = int(

        pd.to_numeric(

            value(
                "random_forest_prediction",
                0
            ),

            errors="coerce"
        )
    )


    latest_xgb_prediction = int(

        pd.to_numeric(

            value(
                "xgboost_prediction",
                0
            ),

            errors="coerce"
        )
    )


    latest_latency = float(

        pd.to_numeric(

            value(
                "latency_ms",
                0
            ),

            errors="coerce"
        )
    )


    latest_agreement = str(

        value(
            "model_agreement",
            True
        )
    ).lower() in [

        "true",
        "1",
        "yes"
    ]


    # ========================================================
    # RETURN LIVE DATA
    # ========================================================

    return {

        "status":
            "active",

        "total_predictions":
            total_predictions,

        "fire_predictions":
            fire_predictions,

        "not_fire_predictions":
            not_fire_predictions,

        "average_fire_probability":
            round(
                average_probability,
                4
            ),

        "average_fire_probability_percent":
            round(
                average_probability * 100,
                2
            ),

        "average_latency_ms":
            round(
                average_latency,
                2
            ),

        "model_agreement_percent":
            round(
                agreement_percent,
                2
            ),

        "latest": {

            "prediction":
                latest_prediction,

            "result":
                str(
                    value(
                        "result",
                        "Fire"
                        if latest_prediction == 1
                        else "Not Fire"
                    )
                ),

            "risk_level":
                str(
                    value(
                        "risk_level",
                        calculate_risk(
                            latest_probability
                        )
                    )
                ),

            "fire_probability":
                round(
                    latest_probability,
                    4
                ),

            "fire_probability_percent":
                round(
                    latest_probability * 100,
                    2
                ),

            "confidence_percent":
                round(
                    float(
                        value(
                            "confidence",
                            latest_probability
                            if latest_prediction == 1
                            else 1 - latest_probability
                        )
                    ) * 100,
                    2
                ),

            "random_forest": {

                "prediction":
                    latest_rf_prediction,

                "probability_percent":
                    round(
                        latest_rf_probability * 100,
                        2
                    )
            },

            "xgboost": {

                "prediction":
                    latest_xgb_prediction,

                "probability_percent":
                    round(
                        latest_xgb_probability * 100,
                        2
                    )
            },

            "model_agreement":
                latest_agreement,

            "latency_ms":
                round(
                    latest_latency,
                    2
                ),

            "timestamp":
                str(
                    value(
                        "timestamp",
                        datetime.now().isoformat()
                    )
                )
        }
    }


# ============================================================
# 14. RECENT PREDICTIONS
# ============================================================

@app.get("/predictions")
def recent_predictions():

    if not os.path.exists(LOG_PATH):

        return {
            "predictions": []
        }


    try:

        logs = pd.read_csv(
            LOG_PATH
        )

    except Exception:

        return {
            "predictions": []
        }


    if logs.empty:

        return {
            "predictions": []
        }


    # Last 10 predictions
    recent = logs.tail(
        10
    ).copy()


    # Replace NaN
    recent = recent.fillna(
        ""
    )


    return {

        "predictions":
            recent.to_dict(
                orient="records"
            )
    }


# ============================================================
# 15. RUN MESSAGE
# ============================================================

print()
print("=" * 70)
print("FORESTIQ API READY")
print("=" * 70)
print("API       : http://127.0.0.1:8000")
print("Docs      : http://127.0.0.1:8000/docs")
print("Health    : http://127.0.0.1:8000/health")
print("Monitoring: http://127.0.0.1:8000/monitoring")
print("=" * 70)
print()