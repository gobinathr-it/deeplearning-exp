"""Ex 9(a) - Implement an LSTM based autoencoder in TensorFlow."""
import numpy as np
from tensorflow.keras.layers import LSTM, Input, RepeatVector
from tensorflow.keras.models import Model

# Set hyperparameters ----------------------------------------------------------
timesteps = 10    # number of timesteps in each sequence
input_dim = 1     # dimensionality of each input element
latent_dim = 64   # dimensionality of the latent space representation
batch_size = 32
epochs = 10

# Define the autoencoder model -------------------------------------------------
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

autoencoder = Model(inputs, decoded)

# Compile the autoencoder ------------------------------------------------------
autoencoder.compile(optimizer="adam", loss="mse")

# Train the autoencoder using random data --------------------------------------
X_train = np.random.random((1000, timesteps, input_dim))
autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs)

# Test the autoencoder on new data ---------------------------------------------
X_test = np.random.random((10, timesteps, input_dim))
decoded_output = autoencoder.predict(X_test)
print("Decoded output shape:", decoded_output.shape)
