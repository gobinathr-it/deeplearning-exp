"""Ex 4(b) - Implement a feed-forward neural network in Keras."""
import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential

# Generate some dummy data -----------------------------------------------------
X = np.random.rand(100, 10)               # 100 samples with 10 features each
y = np.random.randint(2, size=100)        # binary labels (0 or 1)

# Create a sequential model ----------------------------------------------------
model = Sequential()
model.add(Input(shape=(10,)))                    # 10 input features
model.add(Dense(units=64, activation="relu"))    # first hidden layer
model.add(Dense(units=32, activation="relu"))    # second hidden layer
model.add(Dense(units=1, activation="sigmoid"))  # output layer

# Compile the model ------------------------------------------------------------
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model --------------------------------------------------------------
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions -------------------------------------------------------------
sample_input = np.random.rand(1, 10)
prediction = model.predict(sample_input)
print("Prediction:", prediction)
