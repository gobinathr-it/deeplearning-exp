# Ex 10 - Image Generation using GAN

## Aim

To implement image generation using a Generative Adversarial Network (GAN) on MNIST digits.

## Algorithm

1. Import the necessary libraries.
2. Create the generator and discriminator model functions.
3. Call the functions to build the models.
4. (Optional) Train the GAN on MNIST.
5. Generate the images from random noise.
6. Plot the images.

## Project Files

| File | Purpose |
| --- | --- |
| `gan_image_generation.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex10_gan_image_generation
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
python gan_image_generation.py
```

## Expected Output

Per-epoch generator/discriminator losses, then `generated_images.png` containing 10 generated digits.
With `--epochs 0` the images are produced by an untrained generator and look like noise.

## Notes

- **Addition to the lab manual:** the original program only builds the networks, so its output is pure noise. This version includes a standard GAN training loop (default 5 epochs) so the output resembles digits. Use `--epochs 0` to reproduce the manual's untrained behaviour.
- Use `--epochs 20` or more for clearer digits; GPU strongly recommended for long runs.
- The image grid is saved to `generated_images.png` for headless environments.

## Result

Thus the image generation using GAN was executed successfully.
