import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# --------------------------------
# 1. Load final cleaned dataset
# --------------------------------

df = pd.read_csv(
    "data/final_air_quality.csv",
    parse_dates=["Timestamp"]
)

print("Dataset shape:", df.shape)

# --------------------------------
# 2. Select PM2.5
# --------------------------------

values = df[["PM2.5"]].values

# --------------------------------
# 3. Scale PM2.5 values
# --------------------------------

scaler = MinMaxScaler()

scaled_values = scaler.fit_transform(values)

print("Scaled data shape:", scaled_values.shape)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# --------------------------------
# 4. Create 24-hour sequences
# --------------------------------

sequence_length = 24

X = []
y = []

for i in range(sequence_length, len(scaled_values)):

    # Previous 24 hours
    X.append(
        scaled_values[i - sequence_length:i]
    )

    # Next hour
    y.append(
        scaled_values[i]
    )

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# --------------------------------
# 5. Display shapes
# --------------------------------

print("\nSequence data created!")

print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------------------
# 6. Save sequences
# --------------------------------

np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("\nFiles saved successfully!")
print("data/X.npy")
print("data/y.npy")
print("models/scaler.pkl")