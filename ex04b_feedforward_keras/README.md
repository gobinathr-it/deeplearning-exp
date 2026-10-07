# Ex 4(b) - Feed-Forward Network in Keras

## Aim

To implement a feed-forward network in Keras.

## Algorithm

1. Import the necessary libraries.
2. Generate the dummy data.
3. Create a sequential model.
4. Compile the model.
5. Train the model.
6. Make a prediction.
7. Stop.

## Project Files

| File | Purpose |
| --- | --- |
| `feedforward_keras.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex04b_feedforward_keras
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
python feedforward_keras.py
```

## Expected Output

```
Prediction: [[0.5...]]
```
The value is a probability between 0 and 1. Because the data is random, accuracy will hover around 50% - this is expected.

## Notes

- Dummy random data is used purely to demonstrate the network structure.

## Result

Thus the feed-forward network in Keras was executed successfully.
