# Ex 6 - Improve Deep Learning by Fine-Tuning Hyperparameters

## Aim

To improve a deep learning model by fine-tuning hyperparameters (grid search over learning rate and dropout rate on MNIST).

## Algorithm

1. Import the necessary libraries.
2. Load the MNIST dataset.
3. Preprocess the data.
4. Define the model (parameterised by learning rate and dropout rate).
5. Define the hyperparameters to tune.
6. Perform grid search for hyperparameter tuning and print the results.

## Project Files

| File | Purpose |
| --- | --- |
| `hyperparameter_tuning.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex06_hyperparameter_tuning
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
python hyperparameter_tuning.py
```

## Expected Output

Six result blocks (3 learning rates x 2 dropout rates), each showing train and test accuracy:

```
Learning Rate: 0.001, Dropout Rate: 0.25
Train Accuracy: ..., Test Accuracy: ...
--------------------------------------------------
```
A learning rate of 0.1 with Adam typically performs far worse than 0.001 - this illustrates why tuning matters.

## Notes

- 6 models x 5 epochs; on CPU this can take 10-20 minutes. Lower `EPOCHS` for a quick run.

## Result

Thus the deep learning model was improved by fine-tuning hyperparameters successfully.
