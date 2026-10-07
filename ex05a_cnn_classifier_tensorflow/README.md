# Ex 5(a) - Image Classifier using CNN in TensorFlow

## Aim

To implement an image classifier using a CNN in TensorFlow (CIFAR-10).

## Algorithm

1. Import the necessary libraries.
2. Load the dataset.
3. Define the model using CNN.
4. Compile the model with an appropriate loss function, optimizer and metrics.
5. Train and evaluate the model.
6. Plot the training and validation accuracy.

## Project Files

| File | Purpose |
| --- | --- |
| `cnn_tf.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex05a_cnn_classifier_tensorflow
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
python cnn_tf.py
```

## Expected Output

10 training epochs, a `Test accuracy:` line (typically ~68-72%), and a generated `accuracy_plot.png`.

## Notes

- CIFAR-10 (~170 MB) is downloaded on first run.
- Training on CPU can take several minutes; reduce `epochs` in the script for a quick test.
- The plot is saved to `accuracy_plot.png` so it works in headless environments.

## Result

Thus the image classifier using CNN in TensorFlow was executed successfully.
