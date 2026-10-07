"""Ex 4(a) - Implement a feed-forward network in TensorFlow (MNIST)."""
import tensorflow as tf

# Load dataset -----------------------------------------------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data ---------------------------------------------------------------
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define model -----------------------------------------------------------------
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

# Define loss function and optimizer -------------------------------------------
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Compile the model ------------------------------------------------------------
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# Train the model --------------------------------------------------------------
model.fit(x_train, y_train, epochs=5)

# Evaluate the model -----------------------------------------------------------
model.evaluate(x_test, y_test, verbose=2)
