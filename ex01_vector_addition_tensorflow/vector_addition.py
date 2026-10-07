"""Ex 1 - Implement simple vector addition in TensorFlow."""
import numpy as np
import tensorflow as tf

# 1. Matrix addition ---------------------------------------------------------
matrix = tf.constant([[1, 2], [3, 4]])
matrix1 = tf.constant([[2, 4], [6, 8]])
print("Matrix addition:\n", (matrix + matrix1).numpy())

# 2. Vector addition ---------------------------------------------------------
vector1 = tf.constant([10, 10])
vector2 = tf.constant([20, 20])
print("Vector addition:", (vector1 + vector2).numpy())

# 3. Python list + scalar (broadcasting) -------------------------------------
x = [1, 2, 3, 4, 5]
y = 1
print("tf.add(list, scalar):", tf.add(x, y).numpy())

# 4. Tensor + tensor ---------------------------------------------------------
x = tf.convert_to_tensor([1, 2, 3, 4, 5])
y = tf.convert_to_tensor(1)
print("x + y:", (x + y).numpy())

# 5. Python list + tensor ----------------------------------------------------
x = [1, 2, 3, 4, 5]
y = tf.constant([1, 2, 3, 4, 5])
print("tf.add(list, tensor):", tf.add(x, y).numpy())

# 6. int8 overflow behaviour -------------------------------------------------
x = tf.constant([1, 2], dtype=tf.int8)
y = [2**7 + 1, 2**7 + 2]
try:
    print("int8 add:", tf.add(x, y).numpy())
except Exception as exc:  # behaviour differs between TF versions
    print("int8 add raised:", type(exc).__name__, "-", exc)

# 7. Broadcasting shapes -----------------------------------------------------
x = np.ones(6).reshape(1, 2, 1, 3)
y = np.ones(6).reshape(2, 1, 3, 1)
print("Broadcast shape (1,2,1,3)+(2,1,3,1):", tf.add(x, y).shape.as_list())

x = np.ones([1, 2, 1, 4])
y = np.ones([3, 4])
print("Broadcast shape (1,2,1,4)+(3,4):    ", tf.add(x, y).shape.as_list())
