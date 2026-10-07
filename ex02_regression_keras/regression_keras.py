"""Ex 2 - Implement a regression model in Keras."""
import os

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# 1/2. Load the data ---------------------------------------------------------
CSV_PATH = "data.csv"

if not os.path.exists(CSV_PATH):
    print(f"{CSV_PATH} not found - generating a synthetic dataset.")
    rng = np.random.default_rng(42)
    features = rng.uniform(-1, 1, size=(500, 3))
    target = 3 * features[:, 0] - 2 * features[:, 1] + 0.5 * features[:, 2] + 4
    target += rng.normal(0, 0.1, size=500)
    np.savetxt(CSV_PATH, np.column_stack([features, target]), delimiter=",")

data = np.loadtxt(CSV_PATH, delimiter=",")
X_train = data[:, :-1]
y_train = data[:, -1]

# 3. Define the model architecture -------------------------------------------
model = Sequential(
    [
        Input(shape=(X_train.shape[1],)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="linear"),
    ]
)

# 4. Compile the model -------------------------------------------------------
model.compile(loss="mean_squared_error", optimizer="adam")

# 5. Train the model ---------------------------------------------------------
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=2)

# 6. Evaluate the model ------------------------------------------------------
mse = model.evaluate(X_train, y_train, verbose=0)
print("MSE: %.4f" % mse)
