import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


# --------------------------------
# 1. Load final dataset
# --------------------------------

df = pd.read_csv(
    "data/final_air_quality.csv",
    parse_dates=["Timestamp"]
)

values = df[["PM2.5"]].values

print("Total records:", len(values))


# --------------------------------
# 2. Chronological 80/20 split
# --------------------------------

split_index = int(len(values) * 0.8)

train_values = values[:split_index]
test_values = values[split_index:]

print("\nTraining records:", len(train_values))
print("Testing records:", len(test_values))


# --------------------------------
# 3. Fit scaler ONLY on training data
# --------------------------------

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_values)

test_scaled = scaler.transform(test_values)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("\nScaler fitted only on training data.")


# --------------------------------
# 4. Create training sequences
# --------------------------------

sequence_length = 24

X_train = []
y_train = []

for i in range(sequence_length, len(train_scaled)):

    X_train.append(
        train_scaled[i - sequence_length:i]
    )

    y_train.append(
        train_scaled[i]
    )


X_train = np.array(X_train)
y_train = np.array(y_train)


# --------------------------------
# 5. Create testing sequences
# --------------------------------

# Use the last 24 training hours as context
# for the beginning of the test period.

test_input = np.concatenate(
    [train_scaled[-sequence_length:], test_scaled]
)

X_test = []
y_test = []

for i in range(sequence_length, len(test_input)):

    X_test.append(
        test_input[i - sequence_length:i]
    )

    y_test.append(
        test_input[i]
    )


X_test = np.array(X_test)
y_test = np.array(y_test)


# --------------------------------
# 6. Display shapes
# --------------------------------

print("\nTraining sequences:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting sequences:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------
# 7. Save datasets
# --------------------------------

np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)

np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)


print("\nCorrected datasets saved successfully!")

print("data/X_train.npy")
print("data/y_train.npy")
print("data/X_test.npy")
print("data/y_test.npy")
print("models/scaler.pkl")