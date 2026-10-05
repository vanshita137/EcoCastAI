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

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


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
# 4. Load training-only scaler
# --------------------------------

scaler = joblib.load("models/scaler.pkl")


# --------------------------------
# 5. Convert back to PM2.5 values
# --------------------------------

y_actual = scaler.inverse_transform(y_test)

y_predicted = scaler.inverse_transform(predictions)


# --------------------------------
# 6. Calculate MAE
# --------------------------------

mae = mean_absolute_error(
    y_actual,
    y_predicted
)


# --------------------------------
# 7. Calculate RMSE
# --------------------------------

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_predicted
    )
)


# --------------------------------
# 8. Display final results
# --------------------------------

print("\n===================================")
print("       FINAL MODEL PERFORMANCE")
print("===================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

print("===================================")


# --------------------------------
# 9. Plot Actual vs Predicted
# --------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    y_actual[:300],
    label="Actual PM2.5"
)

plt.plot(
    y_predicted[:300],
    label="Predicted PM2.5"
)

plt.title("Actual vs Predicted PM2.5")

plt.xlabel("Time")

plt.ylabel("PM2.5")

plt.legend()

plt.tight_layout()

plt.show()