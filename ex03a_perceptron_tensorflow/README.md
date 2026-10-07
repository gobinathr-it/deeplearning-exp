# Ex 3(a) - Perceptron in TensorFlow

## Aim

To implement a perceptron in TensorFlow.

## Algorithm

1. Import the necessary libraries.
2. Define the perceptron function.
3. Define the training data (AND gate with labels -1 / +1).
4. Define the perceptron variables (weights and bias).
5. Define the perceptron training process.
6. Test the perceptron's predictions.

## Project Files

| File | Purpose |
| --- | --- |
| `perceptron_tf.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex03a_perceptron_tensorflow
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
python perceptron_tf.py
```

## Expected Output

```
Epoch: 1  Loss: <value>
...
Predictions: [[-1.] [-1.] [-1.] [ 1.]]
Accuracy: 1.0
```

## Notes

- **Change from the lab manual:** `tf.math.sign` has a zero gradient, so training through it never updates the weights. The script trains with a smooth `tanh` activation and uses `sign` only for the final prediction.
- Learning rate and epochs were raised (0.5 / 200) so the AND gate converges reliably.

## Result

Thus the perceptron in TensorFlow was executed successfully.
