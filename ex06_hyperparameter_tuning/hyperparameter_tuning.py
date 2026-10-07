"""Ex 6 - Improve deep learning by fine-tuning hyperparameters (grid search)."""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

EPOCHS = 5
BATCH_SIZE = 64

# Load MNIST dataset -----------------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data ----------------------------------------------------------
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# Define the model -------------------------------------------------------------
def build_model(learning_rate=0.001, dropout_rate=0.25):
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Hyperparameters to tune ------------------------------------------------------
learning_rates = [0.001, 0.01, 0.1]
dropout_rates = [0.25, 0.5]

# Perform grid search for hyperparameter tuning --------------------------------
for lr in learning_rates:
    for dr in dropout_rates:
        model = build_model(learning_rate=lr, dropout_rate=dr)
        history = model.fit(
            train_images,
            train_labels,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(test_images, test_labels),
            verbose=0,
        )
        # Print results
        print(f"Learning Rate: {lr}, Dropout Rate: {dr}")
        print(
            f"Train Accuracy: {history.history['accuracy'][-1]}, "
            f"Test Accuracy: {history.history['val_accuracy'][-1]}"
        )
        print("-" * 50)
