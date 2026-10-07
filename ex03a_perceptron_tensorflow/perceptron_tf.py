"""Ex 3(a) - Implement a perceptron in TensorFlow (AND gate)."""
import tensorflow as tf


# Define the perceptron function ---------------------------------------------
def perceptron(X, weights, bias):
    linear = tf.matmul(X, weights) + bias  # linear combination
    return tf.math.sign(linear)            # step-like activation (prediction)


# Define training data -------------------------------------------------------
X_train = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
y_train = tf.constant([[-1], [-1], [-1], [1]], dtype=tf.float32)

# Define the perceptron's variables ------------------------------------------
weights = tf.Variable(tf.zeros([2, 1]), dtype=tf.float32)
bias = tf.Variable(tf.zeros([1]), dtype=tf.float32)

# Define the perceptron's training process -----------------------------------
learning_rate = 0.5
epochs = 200

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # tanh is a differentiable stand-in for sign() during training
        y_soft = tf.tanh(tf.matmul(X_train, weights) + bias)
        loss = tf.reduce_mean(tf.square(y_train - y_soft))
    gradients = tape.gradient(loss, [weights, bias])
    weights.assign_sub(learning_rate * gradients[0])
    bias.assign_sub(learning_rate * gradients[1])
    if epoch == 0 or (epoch + 1) % 20 == 0:
        print("Epoch:", epoch + 1, " Loss:", loss.numpy())

# Test the perceptron's predictions ------------------------------------------
X_test = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
y_test = tf.constant([[-1], [-1], [-1], [1]], dtype=tf.float32)
y_pred = perceptron(X_test, weights, bias)
print("Predictions:", y_pred.numpy().tolist())
print("Accuracy:", float(tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), tf.float32))))
