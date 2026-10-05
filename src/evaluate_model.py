import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------
# 1. Load test data
# --------------------------------

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# --------------------------------
# 2. Load trained model
# --------------------------------

model = load_model("models/lstm_air_quality.keras")


# --------------------------------
# 3. Make predictions
# --------------------------------

predictions = model.predict(X_test)

print("\nPrediction completed!")


# --------------------------------
# 4. Load scaler
# --------------------------------

scaler = joblib.load("models/scaler.pkl")


# Convert predictions back to original PM2.5 values
y_test_original = scaler.inverse_transform(y_test)
predictions_original = scaler.inverse_transform(predictions)


# --------------------------------
# 5. Calculate MAE
# --------------------------------

mae = mean_absolute_error(
    y_test_original,
    predictions_original
)


# --------------------------------
# 6. Calculate RMSE
# --------------------------------

rmse = np.sqrt(
    mean_squared_error(
        y_test_original,
        predictions_original
    )
)


# --------------------------------
# 7. Display results
# --------------------------------

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.2f}")

print(f"RMSE : {rmse:.2f}")

print("=======================================")


# --------------------------------
# 8. Plot Actual vs Predicted
# --------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    y_test_original[:300],
    label="Actual PM2.5"
)

plt.plot(
    predictions_original[:300],
    label="Predicted PM2.5"
)

plt.title("Actual vs Predicted PM2.5")

plt.xlabel("Time")

plt.ylabel("PM2.5")

plt.legend()

plt.tight_layout()

plt.show()