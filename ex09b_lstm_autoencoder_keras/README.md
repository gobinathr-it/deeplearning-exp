# Ex 9(b) - LSTM-Based Autoencoder in Keras

## Aim

To implement an LSTM-based autoencoder in Keras.

## Algorithm

1. Set hyperparameters.
2. Define the autoencoder model.
3. Compile the autoencoder.
4. Train the autoencoder using random data.
5. Test the autoencoder on new data.

## Project Files

| File | Purpose |
| --- | --- |
| `lstm_autoencoder_keras.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex09b_lstm_autoencoder_keras
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
python lstm_autoencoder_keras.py
```

## Expected Output

10 training epochs with a decreasing `loss`, then:

```
Decoded output shape: (10, 10, 1)
```

## Notes

- Same architecture as Ex 9(a), written against the standalone `keras` package imports.

## Result

Thus the LSTM-based autoencoder in Keras was executed successfully.
