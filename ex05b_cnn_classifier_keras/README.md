# Ex 5(b) - Image Classifier using CNN in Keras

## Aim

To implement an image classifier using a CNN in Keras (CIFAR-10).

## Algorithm

1. Import the necessary libraries.
2. Load the dataset.
3. Define the model using CNN.
4. Compile the model with an appropriate loss function, optimizer and metrics.
5. Train and evaluate the model.

## Project Files

| File | Purpose |
| --- | --- |
| `cnn_keras.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex05b_cnn_classifier_keras
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
python cnn_keras.py
```

## Expected Output

10 training epochs followed by the test accuracy value (typically ~68-72%).

## Notes

- CIFAR-10 (~170 MB) is downloaded on first run.
- Training on CPU can take several minutes; reduce `epochs` for a quick test.

## Result

Thus the image classifier using CNN in Keras was executed successfully.
