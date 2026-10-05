import pandas as pd

INPUT_FILE = "data/cleaned_air_quality.csv"
OUTPUT_FILE = "data/final_air_quality.csv"

# Load data
df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["Timestamp"]
)

# Set Timestamp as index
df = df.set_index("Timestamp")

# Create continuous hourly timeline
df = df.asfreq("h")

print("Records after creating hourly timeline:", len(df))

# Check missing PM2.5 values
print("\nMissing PM2.5 before interpolation:")
print(df["PM2.5"].isna().sum())

# Fill missing PM2.5 using time interpolation
df["PM2.5"] = df["PM2.5"].interpolate(method="time")

# Check again
print("\nMissing PM2.5 after interpolation:")
print(df["PM2.5"].isna().sum())

# Save final dataset
df.to_csv(OUTPUT_FILE)

print("\nFinal dataset saved to:")
print(OUTPUT_FILE)