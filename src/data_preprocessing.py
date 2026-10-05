import pandas as pd

# File paths
INPUT_FILE = "data/air-quality-india.csv"
OUTPUT_FILE = "data/cleaned_air_quality.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort data by time
df = df.sort_values("Timestamp")

# Remove duplicate timestamps if any
df = df.drop_duplicates(subset="Timestamp")

# Keep only the columns we need
df = df[["Timestamp", "PM2.5"]]

# Set Timestamp as index
df = df.set_index("Timestamp")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv(OUTPUT_FILE)

print("\nCleaned shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nCleaned dataset saved to:")
print(OUTPUT_FILE)