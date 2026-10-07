# Ex 7 - Transfer Learning in Image Classification

## Aim

To implement the transfer learning concept in image classification (MobileNetV2 on CIFAR-10).

## Algorithm

1. Import the necessary libraries.
2. Load the CIFAR-10 dataset.
3. Preprocess the data.
4. One-hot encode the labels.
5. Load the pre-trained model (MobileNetV2, ImageNet weights).
6. Freeze the convolutional base.
7. Build the new classification model on top of the base.
8. Train and evaluate the model.

## Project Files

| File | Purpose |
| --- | --- |
| `transfer_learning.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex07_transfer_learning
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
python transfer_learning.py
```

## Expected Output

10 training epochs followed by `Test accuracy: <value>`. Accuracy is well above the 10% random baseline; exact value depends on hardware and seed.

## Notes

- Internet access is required on first run to download the ImageNet weights and CIFAR-10.
- **Change from the lab manual:** images are upscaled from 32x32 to 96x96 inside the model (`Resizing` layer), because MobileNetV2 ImageNet weights are not defined for 32x32 inputs. This materially improves accuracy.
- Optional fine-tuning (unfreezing the base with a low learning rate) is available via the `--fine-tune` flag.
- Use `--epochs N` to shorten training on slow machines, e.g. `python transfer_learning.py --epochs 2`.

## Result

Thus the transfer learning concept in image classification was implemented successfully.
