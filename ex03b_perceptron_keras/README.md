# Ex 3(b) - Perceptron in Keras

## Aim

To implement a perceptron in Keras.

## Algorithm

1. Import the necessary libraries.
2. Create a sequential model.
3. Add a single dense layer with a single output and input dimension of 2.
4. Add an activation layer with a sigmoid activation function.
5. Compile the model.
6. Train and evaluate the model.

## Project Files

| File | Purpose |
| --- | --- |
| `perceptron_keras.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex03b_perceptron_keras
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
python perceptron_keras.py
```

## Expected Output

```
Accuracy:  1.0
```
(AND is linearly separable; if you see a lower value, increase `epochs`.)

## Notes

- Imports were updated for Keras 3 (`keras.layers.core` no longer exists).

## Result

Thus the perceptron in Keras was executed successfully.
