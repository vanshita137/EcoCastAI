import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model


# --------------------------------
# 1. Load model and scaler
# --------------------------------

model = load_model("models/lstm_air_quality.keras")

scaler = joblib.load("models/scaler.pkl")


# --------------------------------
# 2. Load final air-quality data
# --------------------------------

df = pd.read_csv(
    "data/final_air_quality.csv",
    parse_dates=["Timestamp"]
)

df = df.sort_values("Timestamp")


# --------------------------------
# 3. Get the latest 24 hours
# --------------------------------

last_24_values = df["PM2.5"].values[-24:]

last_24_values = last_24_values.reshape(-1, 1)


# Scale the values
current_sequence = scaler.transform(last_24_values)


# --------------------------------
# 4. Forecast next 24 hours
# --------------------------------

future_predictions = []

sequence = current_sequence.copy()

for i in range(24):

    # Reshape for LSTM
    X_input = sequence.reshape(1, 24, 1)

    # Predict next hour
    prediction = model.predict(
        X_input,
        verbose=0
    )

    # Save prediction
    future_predictions.append(
        prediction[0, 0]
    )

    # Add prediction to sequence
    sequence = np.append(
        sequence[1:],
        [[prediction[0, 0]]],
        axis=0
    )


# --------------------------------
# 5. Convert predictions back
# --------------------------------

future_predictions = np.array(
    future_predictions
).reshape(-1, 1)

future_predictions = scaler.inverse_transform(
    future_predictions
).flatten()


# --------------------------------
# 6. Create future timestamps
# --------------------------------

last_timestamp = df["Timestamp"].iloc[-1]

future_timestamps = pd.date_range(
    start=last_timestamp + pd.Timedelta(hours=1),
    periods=24,
    freq="h"
)


# --------------------------------
# 7. Create forecast table
# --------------------------------

forecast_df = pd.DataFrame({
    "Timestamp": future_timestamps,
    "Predicted_PM2.5": future_predictions
})


# --------------------------------
# 8. Display forecast
# --------------------------------

print("\n===================================")
print("       NEXT 24 HOURS FORECAST")
print("===================================")

print(forecast_df.to_string(index=False))

print("===================================")


# --------------------------------
# 9. Save forecast
# --------------------------------

forecast_df.to_csv(
    "data/forecast_24_hours.csv",
    index=False
)

print("\nForecast saved to:")
print("data/forecast_24_hours.csv")