# SMS Spam Detection with an LSTM (TECH 405 RNN Hackathon)

**Dataset:** #1 SMS Spam Collection (spam / ham, 2 classes) - baseline 86.6%, target 96%.
**Model:** `nn.Embedding -> nn.LSTM -> nn.Linear` (no pretrained models).

## Why this is a sequence problem
An SMS is an ordered sequence of words. Meaning depends on word order and context
(e.g. "free entry" vs "entry is free for members"), and messages vary in length.
An LSTM reads the tokens one step at a time, carrying a hidden state, and the final
hidden state summarises the message for the classifier.

## How to run
```bash
pip install -r requirements.txt
python train.py          # downloads spam.csv automatically if missing
```
Works on a laptop CPU in about a minute, or in Google Colab (upload `train.py`, run `!python train.py`).
Everything is written to `outputs/`.

## Experimental setup
| Item | Value |
|---|---|
| Split | fixed 80/20 train/test (`random_state=42`, stratified) |
| Validation | 12.5% of the *training* part (model selection only; test set never used for tuning) |
| Tokenisation | lowercase regex word/punctuation tokens, `MAXLEN=50`, words seen < 2 times -> `<unk>` |
| Model | Embedding(64) -> LSTM(128, 1 layer, packed sequences) -> Linear(2) |
| Training | Adam lr 2e-3, batch 64, 12 epochs, gradient clipping 1.0, cross-entropy |

## Report (fill in from your own run, then add screenshots)

### Snapshot 1 - data and shapes
*Screenshot of the top of `outputs/log.txt` / terminal: first rows, class counts, vocab size, batch shapes.*
`![data](screenshots/1_data.png)`
Caption: _SMS Spam data with class counts and the (batch, MAXLEN) tensor shape fed to the LSTM._

### Snapshot 2 - model definition
*Screenshot of the `SpamLSTM` class in `train.py` plus the printed model and parameter count.*
`![model](screenshots/2_model.png)`
Caption: _Embedding, single-layer LSTM and linear classifier._

### Snapshot 3 - training loss decreasing
*Screenshot of the epoch log, and `outputs/loss_curve.png`.*
`![loss](outputs/loss_curve.png)`
Caption: _Training and validation loss per epoch; validation accuracy on the right._

### Snapshot 4 - final test accuracy and plot
*Screenshot of the "final test result" block, and `outputs/confusion_matrix.png`.*
`![cm](outputs/confusion_matrix.png)`
Caption: _Test confusion matrix; final test accuracy shown in the title._

### Result vs baseline and target
Test accuracy **XX.XX%** vs baseline 86.6% and target 96% (copy the value from `outputs/results.json`).

### What worked and what didn't
One honest sentence, e.g. which change (packed sequences, `MIN_FREQ`, learning rate, epochs) helped, and what
the errors look like (use the confusion matrix: are you missing spam, or flagging real messages?).

## Publish to GitHub
```bash
git init
git add train.py requirements.txt README.md outputs screenshots
git commit -m "LSTM SMS spam classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/rnn-sms-spam.git
git push -u origin main
```
Submit the repository URL with your report.
