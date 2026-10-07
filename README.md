# Deep Learning Laboratory - Experiments 1 to 10

Self-contained implementations of all ten laboratory experiments. Each experiment lives in its own folder with its program, `requirements.txt` and a `README.md` containing the aim, algorithm, setup, run instructions, expected output and result.

## Repository Layout

```
dl-lab/
|-- README.md
|-- requirements.txt        # combined dependencies for all experiments
|-- run_all.py              # optional helper to run every experiment
|-- ex01_vector_addition_tensorflow/
|   |-- vector_addition.py
|   |-- requirements.txt
|   `-- README.md
`-- ... (one folder per experiment)
```

## Experiments

| # | Folder | Experiment | Program |
| --- | --- | --- | --- |
| 1 | [`ex01_vector_addition_tensorflow`](./ex01_vector_addition_tensorflow) | Simple Vector Addition in TensorFlow | `vector_addition.py` |
| 2 | [`ex02_regression_keras`](./ex02_regression_keras) | Regression Model in Keras | `regression_keras.py` |
| 3 | [`ex03a_perceptron_tensorflow`](./ex03a_perceptron_tensorflow) | Perceptron in TensorFlow | `perceptron_tf.py` |
| 4 | [`ex03b_perceptron_keras`](./ex03b_perceptron_keras) | Perceptron in Keras | `perceptron_keras.py` |
| 5 | [`ex04a_feedforward_tensorflow`](./ex04a_feedforward_tensorflow) | Feed-Forward Network in TensorFlow | `feedforward_tf.py` |
| 6 | [`ex04b_feedforward_keras`](./ex04b_feedforward_keras) | Feed-Forward Network in Keras | `feedforward_keras.py` |
| 7 | [`ex05a_cnn_classifier_tensorflow`](./ex05a_cnn_classifier_tensorflow) | Image Classifier using CNN in TensorFlow | `cnn_tf.py` |
| 8 | [`ex05b_cnn_classifier_keras`](./ex05b_cnn_classifier_keras) | Image Classifier using CNN in Keras | `cnn_keras.py` |
| 9 | [`ex06_hyperparameter_tuning`](./ex06_hyperparameter_tuning) | Improve Deep Learning by Fine-Tuning Hyperparameters | `hyperparameter_tuning.py` |
| 10 | [`ex07_transfer_learning`](./ex07_transfer_learning) | Transfer Learning in Image Classification | `transfer_learning.py` |
| 11 | [`ex08_sentiment_analysis_rnn`](./ex08_sentiment_analysis_rnn) | Sentiment Analysis using RNN | `sentiment_rnn.py` |
| 12 | [`ex09a_lstm_autoencoder_tensorflow`](./ex09a_lstm_autoencoder_tensorflow) | LSTM-Based Autoencoder in TensorFlow | `lstm_autoencoder_tf.py` |
| 13 | [`ex09b_lstm_autoencoder_keras`](./ex09b_lstm_autoencoder_keras) | LSTM-Based Autoencoder in Keras | `lstm_autoencoder_keras.py` |
| 14 | [`ex10_gan_image_generation`](./ex10_gan_image_generation) | Image Generation using GAN | `gan_image_generation.py` |

## Quick Start (all experiments)

```bash
cd dl-lab
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run a single experiment:

```bash
cd ex04a_feedforward_tensorflow
python feedforward_tf.py
```

Or use the helper:

```bash
python run_all.py          # every experiment (long: several downloads and trainings)
python run_all.py ex04     # only experiments whose folder starts with 'ex04'
```

## Environment Requirements

- Python 3.9 - 3.12
- TensorFlow 2.16 or newer (ships with Keras 3)
- Internet access on first run for datasets (MNIST, CIFAR-10, IMDB) and ImageNet weights (Ex 7)
- Approximate disk usage for cached datasets: under 500 MB

## Running in an Online Code Studio / Notebook

If your code studio is Colab/Kaggle-style, TensorFlow is usually pre-installed. Then run only:

```bash
pip install matplotlib numpy
python <program>.py
```

Figures are saved as PNG files (Ex 5(a), Ex 10) instead of opened in a window so they work on headless servers.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: tensorflow` | `pip install -r requirements.txt` inside the active virtual environment |
| `No matching distribution for tensorflow` | Use Python 3.9 - 3.12 |
| Slow training | Lower the `epochs` value in the script, or use a GPU runtime |
| Dataset download fails | Check internet access / proxy settings |

## Deviations from the Lab Manual

Updated for current TensorFlow/Keras releases; algorithm and structure are unchanged.

- Ex 2: supplies synthetic `data.csv` when none exists.
- Ex 3(a): trains with a differentiable `tanh` (the `sign` gradient is zero, so the manual's version never learns).
- Ex 3(b), 4(b), 8, 9: Keras 3 compatible imports; `Input(...)` layer replaces `input_dim` / `input_shape` arguments.
- Ex 7: images are resized to 96x96 for the ImageNet-pretrained MobileNetV2.
- Ex 10: optional GAN training loop added (set `--epochs 0` for the manual's untrained behaviour).

## References

- TensorFlow API: https://www.tensorflow.org/api_docs/python/tf
- Keras documentation: https://keras.io/api/
