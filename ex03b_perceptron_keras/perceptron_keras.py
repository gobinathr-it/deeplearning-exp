"""Ex 3(b) - Implement a perceptron in Keras (AND gate)."""
import numpy as np
from keras.layers import Activation, Dense, Input
from keras.models import Sequential

# Create a sequential model ----------------------------------------------------
model = Sequential()
model.add(Input(shape=(2,)))        # input dimension of 2
model.add(Dense(units=1))           # single dense layer, single output
model.add(Activation("sigmoid"))    # sigmoid activation layer

# Compile the model ------------------------------------------------------------
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Train the model on sample data -----------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
Y = np.array([0, 0, 0, 1], dtype="float32")
model.fit(X, Y, epochs=1000, batch_size=4, verbose=0)

# Evaluate the model -----------------------------------------------------------
loss, accuracy = model.evaluate(X, Y, verbose=0)
print("Accuracy: ", accuracy)
