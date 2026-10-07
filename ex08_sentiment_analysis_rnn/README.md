# Ex 8 - Sentiment Analysis using RNN

## Aim

To perform sentiment analysis on the IMDB reviews dataset using a Recurrent Neural Network.

## Algorithm

1. Set hyperparameters.
2. Load the dataset.
3. Pad sequences to make them the same length.
4. Define the model.
5. Compile the model.
6. Train and evaluate the model.

## Project Files

| File | Purpose |
| --- | --- |
| `sentiment_rnn.py` | Experiment program |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Prerequisites

- Python 3.9 - 3.12 (required by TensorFlow 2.16+)
- `pip` and (recommended) a virtual environment

## Setup

```bash
cd ex08_sentiment_analysis_rnn
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
python sentiment_rnn.py
```

## Expected Output

10 training epochs followed by:

```
Test loss: <value>, Test accuracy: <value>
```
SimpleRNN accuracy on IMDB is typically in the 75-85% range.

## Notes

- The IMDB dataset (~80 MB) is downloaded on first run.
- `keras.preprocessing.sequence` was replaced by `keras.utils.pad_sequences` for Keras 3 compatibility.

## Result

Thus the sentiment analysis using RNN was executed successfully.
