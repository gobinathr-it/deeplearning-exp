# Ex 9(a) - LSTM-Based Autoencoder in TensorFlow

## Aim

To implement an LSTM-based autoencoder in TensorFlow.

## Algorithm

1. Set hyperparameters.
2. Define the autoencoder model.
3. Compile the autoencoder.
4. Train the autoencoder using random data.
5. Test the autoencoder on new data.

## Project Files

| File | Purpose |
| --- | --- |
| `lstm_autoencoder_tf.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex09a_lstm_autoencoder_tensorflow
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
python lstm_autoencoder_tf.py
```

## Expected Output

10 training epochs with a decreasing `loss`, then the shape of the reconstructed output:

```
Decoded output shape: (10, 10, 1)
```

## Notes

- Random data is used for demonstration; reconstruction quality on pure noise is intentionally limited.

## Result

Thus the LSTM-based autoencoder in TensorFlow was executed successfully.
