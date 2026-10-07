"""Ex 10 - Image generation using GAN (MNIST)."""
import argparse

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; the figure is saved to a PNG file
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="0 = skip training")
parser.add_argument("--batch-size", type=int, default=256)
args = parser.parse_args()

NOISE_DIM = 100


# Create model functions -------------------------------------------------------
def build_generator():
    return keras.Sequential(
        [
            layers.Input(shape=(NOISE_DIM,)),
            layers.Dense(7 * 7 * 256),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", activation="tanh"),
        ]
    )


def build_discriminator():
    return keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


def build_gan(generator, discriminator):
    discriminator.trainable = False
    return keras.Sequential([generator, discriminator])


# Call the functions -----------------------------------------------------------
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Optional training loop -------------------------------------------------------
if args.epochs > 0:
    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") - 127.5) / 127.5  # scale to [-1, 1]
    x_train = np.expand_dims(x_train, -1)
    dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(60000)
        .batch(args.batch_size, drop_remainder=True)
    )

    bce = keras.losses.BinaryCrossentropy()
    g_opt = keras.optimizers.Adam(1e-4)
    d_opt = keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(real_images):
        noise = tf.random.normal([tf.shape(real_images)[0], NOISE_DIM])
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = generator(noise, training=True)
            real_out = discriminator(real_images, training=True)
            fake_out = discriminator(fake_images, training=True)
            g_loss = bce(tf.ones_like(fake_out), fake_out)
            d_loss = bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
        d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        return g_loss, d_loss

    for epoch in range(args.epochs):
        for batch in dataset:
            g_loss, d_loss = train_step(batch)
        print(f"Epoch {epoch + 1}/{args.epochs}  generator loss: {float(g_loss):.4f}  "
              f"discriminator loss: {float(d_loss):.4f}")

# Generate the images ----------------------------------------------------------
num_images = 10
noise = np.random.normal(0, 1, (num_images, NOISE_DIM))
generated_images = generator.predict(noise)

# Plot the images --------------------------------------------------------------
plt.figure(figsize=(15, 2))
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(generated_images[i, :, :, 0], cmap="gray")
    plt.axis("off")
plt.savefig("generated_images.png", dpi=120, bbox_inches="tight")
print("Generated images saved to generated_images.png")
