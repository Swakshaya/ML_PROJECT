import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)


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
# 3. CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 4. RANDOM FOREST CROSS-VALIDATION
# ==========================================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_cv_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc"
)

print("=" * 60)
print("RANDOM FOREST CROSS-VALIDATION")
print("=" * 60)

print("Fold ROC-AUC scores:")
print(rf_cv_scores)

print(
    f"\nMean ROC-AUC: "
    f"{rf_cv_scores.mean():.4f}"
)

print(
    f"Std ROC-AUC: "
    f"{rf_cv_scores.std():.4f}"
)


# ==========================================
# 5. XGBOOST CROSS-VALIDATION
# ==========================================

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_cv_scores = cross_val_score(
    xgb,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc"
)

print("\n")
print("=" * 60)
print("XGBOOST CROSS-VALIDATION")
print("=" * 60)

print("Fold ROC-AUC scores:")
print(xgb_cv_scores)

print(
    f"\nMean ROC-AUC: "
    f"{xgb_cv_scores.mean():.4f}"
)

print(
    f"Std ROC-AUC: "
    f"{xgb_cv_scores.std():.4f}"
)


# ==========================================
# 6. RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================

rf_params = {

    "n_estimators": [
        100,
        200
    ],

    "max_depth": [
        4,
        6,
        8,
        None
    ],

    "min_samples_split": [
        2,
        5
    ]
}


rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),

    param_grid=rf_params,

    cv=cv,

    scoring="roc_auc",

    n_jobs=-1
)


rf_grid.fit(
    X_train,
    y_train
)


print("\n")
print("=" * 60)
print("RANDOM FOREST GRID SEARCH")
print("=" * 60)

print(
    "Best parameters:"
)

print(
    rf_grid.best_params_
)

print(
    f"\nBest CV ROC-AUC: "
    f"{rf_grid.best_score_:.4f}"
)


# ==========================================
# 7. XGBOOST HYPERPARAMETER TUNING
# ==========================================

xgb_params = {

    "n_estimators": [
        100,
        200
    ],

    "max_depth": [
        2,
        3,
        4
    ],

    "learning_rate": [
        0.03,
        0.05,
        0.1
    ]
}


xgb_grid = GridSearchCV(

    estimator=XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ),

    param_grid=xgb_params,

    cv=cv,

    scoring="roc_auc",

    n_jobs=-1
)


xgb_grid.fit(
    X_train,
    y_train
)


print("\n")
print("=" * 60)
print("XGBOOST GRID SEARCH")
print("=" * 60)

print(
    "Best parameters:"
)

print(
    xgb_grid.best_params_
)

print(
    f"\nBest CV ROC-AUC: "
    f"{xgb_grid.best_score_:.4f}"
)


# ==========================================
# 8. FINAL MODEL EVALUATION
# ==========================================

models = {

    "Random Forest": rf_grid.best_estimator_,

    "XGBoost": xgb_grid.best_estimator_

}


results = []


for name, model in models.items():

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC-AUC": roc_auc,

        "PR-AUC": pr_auc

    })


# ==========================================
# 9. PRINT FINAL RESULTS
# ==========================================

results_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 80)
print("FINAL TEST SET RESULTS")
print("=" * 80)

print(
    results_df.round(4).to_string(
        index=False
    )
)


# ==========================================
# 10. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "results/final_model_evaluation.csv",
    index=False
)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

best_model = xgb_grid.best_estimator_

y_pred = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Fire",
        "Fire"
    ]
)

disp.plot()

plt.title(
    "XGBoost Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "results/xgboost_confusion_matrix.png",
    dpi=300
)

plt.show()


# ==========================================
# 12. ROC CURVE
# ==========================================

y_prob = best_model.predict_proba(
    X_test
)[:, 1]

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "XGBoost ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/xgboost_roc_curve.png",
    dpi=300
)

plt.show()


# ==========================================
# 13. PRECISION-RECALL CURVE
# ==========================================

precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall,
    precision,
    label=f"PR-AUC = {pr_auc:.3f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "XGBoost Precision-Recall Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/xgboost_precision_recall_curve.png",
    dpi=300
)

plt.show()


print("\n")
print("=" * 60)
print("CO5 COMPLETE")
print("=" * 60)

print("\nSaved:")
print("results/final_model_evaluation.csv")
print("results/xgboost_confusion_matrix.png")
print("results/xgboost_roc_curve.png")
print("results/xgboost_precision_recall_curve.png")