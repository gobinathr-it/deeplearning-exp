# Ex 1 - Simple Vector Addition in TensorFlow

## Aim

To implement simple vector addition in TensorFlow.

## Algorithm

1. Import the necessary libraries.
2. Define the input vectors as tensors with the desired shape and data type.
3. Add the input vectors using the `add` function (or the `+` operator).
4. Run the computation (TensorFlow 2.x uses eager execution, so no explicit session is needed).
5. Print the sum of the input vectors.

## Project Files

| File | Purpose |
| --- | --- |
| `vector_addition.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex01_vector_addition_tensorflow
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python vector_addition.py
```

## Expected Output

```
Matrix addition: [[ 3  6] [ 9 12]]
Vector addition: [30 30]
tf.add(list, scalar): [2 3 4 5 6]
...
Broadcast shape (1,2,1,3)+(2,1,3,1): [2, 2, 3, 3]
Broadcast shape (1,2,1,4)+(3,4):     [1, 2, 3, 4]
```

## Notes

- In TensorFlow 2.x the 'Initialize the session' step of the lab manual is not required; operations execute eagerly.
- The `int8` example demonstrates integer overflow/conversion behaviour; it is wrapped in `try/except` because the result depends on the TensorFlow version.

## Result

Thus the simple vector addition in TensorFlow was executed successfully.
