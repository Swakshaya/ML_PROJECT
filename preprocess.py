import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD RAW DATASET
# ==========================================

file_path = "data/Algerian_forest_fires_dataset_UPDATE.csv"

# The UCI UPDATE file has 1 title row before the actual header.
df = pd.read_csv(
    file_path,
    skiprows=1
)

print("=" * 60)
print("FOREST FIRE DATASET PREPROCESSING")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())

# ==========================================
# 2. CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.astype(str).str.strip()

# ==========================================
# 3. REMOVE EMPTY ROWS
# ==========================================

df = df.dropna(how="all")

# ==========================================
# 4. REMOVE REGION SEPARATOR ROW
# ==========================================

# The original dataset contains a row separating
# the two Algerian regions.

if "day" in df.columns:
    df["day"] = pd.to_numeric(df["day"], errors="coerce")

df = df.dropna(subset=["day"])

# ==========================================
# 5. CONVERT NUMERIC FEATURES
# ==========================================

numeric_columns = [
    "day",
    "month",
    "year",
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ==========================================
# 6. CLEAN TARGET COLUMN
# ==========================================

if "Classes" not in df.columns:
    raise ValueError(
        f"Classes column not found. Columns are: {df.columns.tolist()}"
    )

df["Classes"] = (
    df["Classes"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert target:
# fire     -> 1
# not fire -> 0

df["Fire"] = df["Classes"].map({
    "fire": 1,
    "not fire": 0
})

# ==========================================
# 7. CHECK TARGET VALUES
# ==========================================

print("\nTarget values found:")
print(df["Classes"].value_counts())

# Remove rows where target could not be converted
df = df.dropna(subset=["Fire"])

df["Fire"] = df["Fire"].astype(int)

# ==========================================
# 8. REMOVE MISSING VALUES
# ==========================================

df = df.dropna()

print("\nFinal cleaned dataset shape:")
print(df.shape)

# ==========================================
# 9. SELECT FEATURES
# ==========================================

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

print("\nFeatures:")
print(features)

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

# ==========================================
# 10. TARGET DISTRIBUTION
# ==========================================

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(y.value_counts(normalize=True) * 100)

# ==========================================
# 11. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 12. SAVE PROCESSED DATASET
# ==========================================

processed_df = X.copy()
processed_df["Fire"] = y

processed_df.to_csv(
    "data/forest_fire_processed.csv",
    index=False
)

print("\nProcessed dataset saved successfully:")
print("data/forest_fire_processed.csv")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)