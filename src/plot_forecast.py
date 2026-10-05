import pandas as pd
import matplotlib.pyplot as plt


# Load forecast data
df = pd.read_csv(
    "data/forecast_24_hours.csv",
    parse_dates=["Timestamp"]
)


# Create graph
plt.figure(figsize=(14, 6))

plt.plot(
    df["Timestamp"],
    df["Predicted_PM2.5"],
    marker="o",
    linewidth=2,
    label="Predicted PM2.5"
)


# Title and labels
plt.title(
    "EcoCast AI - Next 24 Hours PM2.5 Forecast",
    fontsize=16
)

plt.xlabel("Time")
plt.ylabel("PM2.5")


# Rotate timestamps
plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()