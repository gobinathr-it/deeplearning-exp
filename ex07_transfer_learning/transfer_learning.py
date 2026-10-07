"""Ex 7 - Implement the transfer learning concept in image classification."""
import argparse

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--fine-tune", action="store_true", help="run optional fine-tuning phase")
args = parser.parse_args()

IMG_SIZE = 96  # MobileNetV2 ImageNet weights need >= 96x96 inputs

# Load CIFAR-10 dataset --------------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocess the data (scales pixels to [-1, 1]) -------------------------------
train_images = preprocess_input(train_images.astype("float32"))
test_images = preprocess_input(test_images.astype("float32"))

# One-hot encode the labels ----------------------------------------------------
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Load pre-trained MobileNetV2 without the top (classification) layers ---------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

# Freeze the convolutional base ------------------------------------------------
base_model.trainable = False

# Build a new model on top of the pre-trained base -----------------------------
model = models.Sequential(
    [
        layers.Input(shape=(32, 32, 3)),
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model ------------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model --------------------------------------------------------------
history = model.fit(
    train_images,
    train_labels,
    epochs=args.epochs,
    validation_data=(test_images, test_labels),
)

# Evaluate the model -----------------------------------------------------------
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Fine-tune the model (optional) -----------------------------------------------
if args.fine_tune:
    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_images,
        train_labels,
        epochs=5,
        validation_data=(test_images, test_labels),
    )
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy after fine-tuning: {test_acc}")
