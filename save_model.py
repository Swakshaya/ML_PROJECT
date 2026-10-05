import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/forest_fire_processed.csv"
)

features = [
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI"
]

X = df[features]
y = df["Fire"]


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. TRAIN FINAL XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# 4. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/forest_fire_xgboost.pkl"
)


# ==========================================
# 5. SAVE FEATURE ORDER
# ==========================================

joblib.dump(
    features,
    "models/features.pkl"
)


print("=" * 60)
print("MODEL PACKAGING COMPLETE")
print("=" * 60)

print("\nModel saved:")
print("models/forest_fire_xgboost.pkl")

print("\nFeatures saved:")
print("models/features.pkl")