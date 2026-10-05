import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------
# 1. Load corrected training data
# --------------------------------

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

print("Training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------
# 2. Build LSTM model
# --------------------------------

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(64, return_sequences=True),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])


# --------------------------------
# 3. Compile model
# --------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# --------------------------------
# 4. Show model architecture
# --------------------------------

model.summary()


# --------------------------------
# 5. Early stopping
# --------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# --------------------------------
# 6. Train model
# --------------------------------

history = model.fit(
    X_train,
    y_train,

    epochs=30,

    batch_size=32,

    validation_split=0.1,

    callbacks=[early_stopping],

    shuffle=False
)


# --------------------------------
# 7. Save model
# --------------------------------

model.save(
    "models/lstm_air_quality.keras"
)

print("\n================================")
print("LSTM TRAINING COMPLETED!")
print("================================")

print("\nModel saved at:")
print("models/lstm_air_quality.keras")