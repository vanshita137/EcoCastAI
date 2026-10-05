import numpy as np

# Load sequence data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# 80% training, 20% testing
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Save the split data
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

print("\nTrain/test data saved successfully!")