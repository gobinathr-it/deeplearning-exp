"""Ex 8 - Perform sentiment analysis using RNN (IMDB)."""
import keras
from keras.datasets import imdb
from keras.layers import Dense, Embedding, Input, SimpleRNN
from keras.models import Sequential
from keras.utils import pad_sequences

# Set hyperparameters ----------------------------------------------------------
max_features = 20000  # number of words to consider as features
maxlen = 100          # maximum length of each sequence
batch_size = 32       # number of samples per gradient update
epochs = 10           # number of epochs to train the model

# Load the dataset -------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to make them the same length -----------------------------------
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Define the model -------------------------------------------------------------
model = Sequential()
model.add(Input(shape=(maxlen,)))
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation="sigmoid"))

# Compile the model ------------------------------------------------------------
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model --------------------------------------------------------------
hist = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
)

# Evaluate the model -----------------------------------------------------------
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Test loss: {loss}, Test accuracy: {acc}")
