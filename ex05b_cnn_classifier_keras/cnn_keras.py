"""Ex 5(b) - Implement an image classifier using CNN in Keras (CIFAR-10)."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset -------------------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Normalize pixel values between 0 and 1 ---------------------------------------
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Define the model using CNN ---------------------------------------------------
model = keras.Sequential(
    [
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)

# Compile the model ------------------------------------------------------------
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model on the training set ------------------------------------------
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
)

# Evaluate the model on the test set -------------------------------------------
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy:", test_acc)
