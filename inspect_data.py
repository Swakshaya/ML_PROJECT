import pandas as pd

# Dataset path
file_path = "data/Algerian_forest_fires_dataset_CLEANED.csv"

# Load dataset
df = pd.read_csv(file_path)

print("=" * 60)
print("FOREST FIRE DATASET INSPECTION")
print("=" * 60)

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Column Names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n3. First 5 Records:")
print(df.head())

print("\n4. Data Types:")
print(df.dtypes)

print("\n5. Missing Values:")
print(df.isnull().sum())

print("\n6. Dataset Information:")
print(df.info())

print("\n7. Statistical Summary:")
print(df.describe())

# Check target column
if "Classes" in df.columns:
    print("\n8. Target Distribution:")
    print(df["Classes"].value_counts())

    print("\nTarget Percentages:")
    print(df["Classes"].value_counts(normalize=True) * 100)
else:
    print("\nWARNING: 'Classes' column was not found.")