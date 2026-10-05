import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv(
    "data/cleaned_air_quality.csv",
    parse_dates=["Timestamp"],
    index_col="Timestamp"
)

# Plot PM2.5
plt.figure(figsize=(14, 5))

plt.plot(df.index, df["PM2.5"])

plt.title("PM2.5 Concentration Over Time")
plt.xlabel("Time")
plt.ylabel("PM2.5")

plt.tight_layout()
plt.show()