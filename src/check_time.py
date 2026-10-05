import pandas as pd

# Load cleaned data
df = pd.read_csv(
    "data/cleaned_air_quality.csv",
    parse_dates=["Timestamp"]
)

# Calculate difference between consecutive timestamps
time_difference = df["Timestamp"].diff()

# Count gaps that are not exactly 1 hour
gaps = time_difference[time_difference != pd.Timedelta(hours=1)]

print("Total records:", len(df))
print("Number of time gaps:", len(gaps))

print("\nTime gaps:")
print(gaps.head(20))