# Ex 4(a) - Feed-Forward Network in TensorFlow

## Aim

To implement a feed-forward network in TensorFlow (MNIST).

## Algorithm

1. Import the necessary libraries.
2. Load the dataset.
3. Normalise the data.
4. Define the model.
5. Define the loss function and optimizer.
6. Compile the model.
7. Train and evaluate the model.

## Project Files

| File | Purpose |
| --- | --- |
| `feedforward_tf.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex04a_feedforward_tensorflow
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
python feedforward_tf.py
```

## Expected Output

5 training epochs followed by an evaluation line. Test accuracy is typically around 97-98%.

## Notes

- The MNIST dataset (~11 MB) is downloaded on first run, so an internet connection is required.

## Result

Thus the feed-forward network in TensorFlow was executed successfully.
