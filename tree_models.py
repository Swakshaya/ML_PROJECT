import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# 1. PATHS
# =========================================================

DATA_PATH = "data/forest_fire_processed.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

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


print("=" * 70)
print("FOREST FIRE TREE-BASED MODEL TRAINING")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nFeatures:")
print(features)

print("\nTarget distribution:")
print(y.value_counts())


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================================================
# 4. DEFINE MODELS
# =========================================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42
    )
}


# =========================================================
# 5. TRAIN AND EVALUATE
# =========================================================

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Train
    model.fit(X_train, y_train)

    trained_models[name] = model

    # Predictions
    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })


# =========================================================
# 6. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results_df.round(4).to_string(index=False)
)


# =========================================================
# 7. SAVE MODEL COMPARISON
# =========================================================

results_df.to_csv(
    f"{RESULTS_DIR}/tree_model_comparison.csv",
    index=False
)


# =========================================================
# 8. SAVE RANDOM FOREST
# =========================================================

joblib.dump(
    trained_models["Random Forest"],
    f"{MODEL_DIR}/forest_fire_random_forest.pkl"
)


# =========================================================
# 9. SAVE XGBOOST
# =========================================================

joblib.dump(
    trained_models["XGBoost"],
    f"{MODEL_DIR}/forest_fire_xgboost.pkl"
)


# =========================================================
# 10. SAVE FEATURE LIST
# =========================================================

joblib.dump(
    features,
    f"{MODEL_DIR}/features.pkl"
)


print("\nModels saved successfully:")

print(
    f"{MODEL_DIR}/forest_fire_random_forest.pkl"
)

print(
    f"{MODEL_DIR}/forest_fire_xgboost.pkl"
)

print(
    f"{MODEL_DIR}/features.pkl"
)


# =========================================================
# 11. RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

rf_model = trained_models["Random Forest"]

rf_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(
    rf_importance.to_string(index=False)
)


rf_importance.to_csv(
    f"{RESULTS_DIR}/random_forest_feature_importance.csv",
    index=False
)


# =========================================================
# 12. XGBOOST FEATURE IMPORTANCE
# =========================================================

xgb_model = trained_models["XGBoost"]

xgb_importance = pd.DataFrame({
    "Feature": features,
    "Importance": xgb_model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 60)

print(
    xgb_importance.to_string(index=False)
)


xgb_importance.to_csv(
    f"{RESULTS_DIR}/xgboost_feature_importance.csv",
    index=False
)


# =========================================================
# 13. RANDOM FOREST GRAPH
# =========================================================

plt.figure(figsize=(10, 6))

plt.barh(
    rf_importance["Feature"],
    rf_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/random_forest_feature_importance.png",
    dpi=300
)

plt.close()


# =========================================================
# 14. XGBOOST GRAPH
# =========================================================

plt.figure(figsize=(10, 6))

plt.barh(
    xgb_importance["Feature"],
    xgb_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/xgboost_feature_importance.png",
    dpi=300
)

plt.close()


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print("models/forest_fire_random_forest.pkl")
print("models/forest_fire_xgboost.pkl")
print("models/features.pkl")

print("\nResults:")

print("results/tree_model_comparison.csv")
print("results/random_forest_feature_importance.csv")
print("results/xgboost_feature_importance.csv")

print("\nFeature graphs generated successfully.")