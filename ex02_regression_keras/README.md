# Ex 2 - Regression Model in Keras

## Aim

To implement a regression model in Keras.

## Algorithm

1. Import the necessary packages and libraries.
2. Load the data.
3. Define the model architecture.
4. Compile the model.
5. Train the model.
6. Evaluate the model.

## Project Files

| File | Purpose |
| --- | --- |
| `regression_keras.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex02_regression_keras
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
python regression_keras.py
```

## Expected Output

Training logs for 100 epochs (`loss` and `val_loss` decreasing) followed by a single line:

```
MSE: <value>
```

## Notes

- The lab manual reads `data.csv`, which is not supplied. If the file is missing, the script **generates a synthetic dataset** (3 features, 1 target) and saves it as `data.csv`.
- To use your own data, place a comma-separated `data.csv` next to the script with the **target in the last column**.

## Result

Thus the regression model in Keras was executed successfully.
