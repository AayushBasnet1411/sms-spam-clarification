"""TECH 405 RNN Hackathon - SMS Spam classification with an LSTM.

Pipeline: spam.csv -> lowercase/tokenise -> vocab -> padded index sequences
          -> nn.Embedding -> nn.LSTM -> nn.Linear -> spam / ham

Run:  python train.py
Outputs (in ./outputs): loss_curve.png, confusion_matrix.png, results.json, log.txt
"""
import json
import os
import re
import urllib.request
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------- config
URL = ("https://raw.githubusercontent.com/mohitgupta-omg/"
       "Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv")
CSV = "spam.csv"
SEED = 42
MAXLEN = 50          # SMS messages are short; 50 tokens covers almost all of them
MIN_FREQ = 2         # words seen fewer times than this become <unk>
EMB, HIDDEN = 64, 128
EPOCHS, BATCH, LR = 12, 64, 2e-3
BASELINE, TARGET = 0.866, 0.96   # from the hackathon sheet

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs("outputs", exist_ok=True)
LOG = open("outputs/log.txt", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# ---------------------------------------------------------------- data
if not os.path.exists(CSV):
    urllib.request.urlretrieve(URL, CSV)
df = pd.read_csv(CSV, encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["y"] = (df["label"] == "spam").astype(int)
log("== Snapshot 1: data ==")
log(df.head())
log("rows:", len(df), "| spam:", int(df.y.sum()), "| ham:", int((1 - df.y).sum()))

tok = lambda t: re.findall(r"[a-z0-9']+|[^\sa-z0-9]", t.lower())

# fixed 80/20 split, stratified so both sets keep the spam ratio
tr_df, te_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.y)
# validation carved out of TRAIN only (never tune on test)
tr_df, va_df = train_test_split(tr_df, test_size=0.125, random_state=SEED, stratify=tr_df.y)

counts = Counter(w for t in tr_df.text for w in tok(t))
itos = ["<pad>", "<unk>"] + [w for w, c in counts.items() if c >= MIN_FREQ]
stoi = {w: i for i, w in enumerate(itos)}


def encode(texts):
    out = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [stoi.get(w, 1) for w in tok(t)][:MAXLEN]
        out[i, :len(ids)] = ids          # right-padded with 0
    return torch.tensor(out)


def loader(d, shuffle):
    ds = TensorDataset(encode(d.text.tolist()), torch.tensor(d.y.values, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


train_dl, val_dl, test_dl = loader(tr_df, True), loader(va_df, False), loader(te_df, False)
log("vocab size:", len(itos))
log("train/val/test:", len(tr_df), len(va_df), len(te_df))
xb, yb = next(iter(train_dl))
log("batch shapes -> X:", tuple(xb.shape), " y:", tuple(yb.shape))
log("test-set majority baseline:", round(1 - te_df.y.mean(), 4))


# ---------------------------------------------------------------- model
class SpamLSTM(nn.Module):
    def __init__(self, vocab, emb, hid):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, 2)

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)     # h: (1, B, hid) = last real step
        return self.fc(h[-1])


dev = "cuda" if torch.cuda.is_available() else "cpu"
model = SpamLSTM(len(itos), EMB, HIDDEN).to(dev)
log("\n== Snapshot 2: model ==")
log(model)
log("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ---------------------------------------------------------------- train
opt = torch.optim.Adam(model.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss()


def run(dl, train=False):
    model.train(train)
    tot_loss = correct = n = 0
    preds, ys = [], []
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = lossf(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot_loss += loss.item() * len(y)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            n += len(y)
            preds += p.cpu().tolist()
            ys += y.cpu().tolist()
    return tot_loss / n, correct / n, ys, preds


log("\n== Snapshot 3: training ==")
hist = {"train_loss": [], "val_loss": [], "val_acc": []}
best_val, best_state = -1, None
for ep in range(1, EPOCHS + 1):
    trl, _, _, _ = run(train_dl, True)
    vl, va, _, _ = run(val_dl)
    hist["train_loss"].append(trl)
    hist["val_loss"].append(vl)
    hist["val_acc"].append(va)
    if va > best_val:                     # model selection uses VALIDATION only
        best_val, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    log(f"epoch {ep:2d}/{EPOCHS}  train_loss {trl:.4f}  val_loss {vl:.4f}  val_acc {va:.4f}")

# ---------------------------------------------------------------- test (once)
model.load_state_dict(best_state)
_, test_acc, ys, ps = run(test_dl)
log("\n== Snapshot 4: final test result ==")
log(f"TEST ACCURACY: {test_acc:.4f}  ({test_acc*100:.2f}%)")
log(f"baseline {BASELINE*100:.1f}%  |  target {TARGET*100:.0f}%  |  "
    f"{'TARGET MET' if test_acc >= TARGET else 'below target'}")
log(classification_report(ys, ps, target_names=["ham", "spam"], digits=4))

# ---------------------------------------------------------------- plots
ep = range(1, EPOCHS + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(ep, hist["train_loss"], marker="o", label="train")
ax[0].plot(ep, hist["val_loss"], marker="o", label="validation")
ax[0].set(title="Training loss", xlabel="epoch", ylabel="cross-entropy")
ax[0].legend()
ax[1].plot(ep, hist["val_acc"], marker="o", color="tab:green")
ax[1].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy")
plt.tight_layout()
plt.savefig("outputs/loss_curve.png", dpi=150)
plt.close()

cm = confusion_matrix(ys, ps)
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.imshow(cm, cmap="Blues")
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["ham", "spam"], yticklabels=["ham", "spam"],
       xlabel="predicted", ylabel="actual", title=f"Test confusion matrix\nacc {test_acc*100:.2f}%")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

json.dump({"test_accuracy": test_acc, "baseline": BASELINE, "target": TARGET,
           "best_val_accuracy": best_val, "epochs": EPOCHS, "vocab": len(itos),
           "history": hist, "confusion_matrix": cm.tolist()},
          open("outputs/results.json", "w"), indent=2)
log("\nSaved outputs/loss_curve.png, outputs/confusion_matrix.png, outputs/results.json")
