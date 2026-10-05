import pandas as pd

# Load dataset
file_path = "data/air-quality-india.csv"

df = pd.read_csv(file_path)

# Show first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# Show dataset size
print("\n--- Dataset Shape ---")
print(df.shape)

# Show column names
print("\n--- Columns ---")
print(df.columns.tolist())

# Show data types
print("\n--- Data Types ---")
print(df.dtypes)

# Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Basic statistics
print("\n--- Statistics ---")
print(df.describe())