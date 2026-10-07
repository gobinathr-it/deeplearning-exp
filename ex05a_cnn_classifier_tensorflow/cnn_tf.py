"""Ex 5(a) - Implement an image classifier using CNN in TensorFlow (CIFAR-10)."""
import matplotlib

matplotlib.use("Agg")  # headless-safe backend; the plot is saved to a PNG file
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load the dataset -------------------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values between 0 and 1 ---------------------------------------
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model using CNN ---------------------------------------------------
model = models.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

# Compile the model ------------------------------------------------------------
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model --------------------------------------------------------------
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
)

# Evaluate the model -----------------------------------------------------------
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy:", test_acc)

# Plot the training and validation accuracy ------------------------------------
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png", dpi=120, bbox_inches="tight")
print("Plot saved to accuracy_plot.png")
