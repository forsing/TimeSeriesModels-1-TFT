#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
1. Temporal Fusion Transformers (TFT) for Multivariate Forecasting

Hybrid forecasting architectures that combine deep learning with classical time-series models. 
These approaches capture complex patterns while retaining interpretability and statistical rigor.
"""


"""
1_TFT_loto_v2.py — Loto 7/39 predikcija iz loto7hh_4620_k41.csv 4620 izvlačenja 
TFT-inspirisan PyTorch model sa selection gate, LSTM, attention, 39 sigmoid izlaza, BEST/FINAL predikcijom i back-testom.

  • TFT sa pravim Loto 7/39 tokom.
  • feature-i: multi-hot, rolling frekvencije 20/50/100, gap, suma, neparni, niski, raspon
  • Ulaz: pravi sekvence iz poslednjih LOOK_BACK = 10 kola.
  • Svaki vremenski korak ima:
      - 39 multi-hot brojeva
      - rolling frekvencije 20/50/100 za brojeve 1..39
      - gap za brojeve 1..39
      - 5 statistika kola: suma, neparni, niski, raspon, prosečan gap
  • Izlaz: 39 sigmoid skorova, pa top-7 jedinstvenih brojeva.
  • Vremenski split: train -> val -> back-test, bez shuffle.
  • Čuvaju se BEST i FINAL težine. BEST, FINAL, ENSEMBLE kombinacije. 
  • Back-test poslednjih 100 kola: hits/7, hit%, AUC, LRAP + slučajan baseline.
  • Predikcija sledećeg kola koristi poslednjih LOOK_BACK kola iz CSV-a.
  • Determinizam: SEED=39, single-thread, PyTorch deterministic.
  • Snima rezultate u 1_TFT_loto_v2_predikcija.txt.
"""


import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Konfiguracija
# =========================
CSV_PATH = "/Users/4c/Desktop/GHQ/KvantniRegresor/loto7hh_4620_k41.csv"
OUT_TXT = Path("/Users/4c/Desktop/GHQ/TimeSeriesModels/TFT_loto_v2_predikcija.txt")

N_MIN, N_MAX = 1, 39
K = 7
LOOK_BACK = 10
WINDOWS = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 462
BATCH = 64
LR = 1e-3
HIDDEN_DIM = 128
DROPOUT = 0.20


T0 = time.time()
print()
print("START 1_TFT_loto_v2", datetime.today())
print()


# =========================
# Učitavanje i validacija CSV-a
# =========================
df = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N = draws.shape[0]

if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV učitan: {CSV_PATH}")
print(f"Broj izvlačenja: {N}, brojeva po kolu: {K}")
print()


def draws_to_multihot(rows: np.ndarray) -> np.ndarray:
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def rolling_features(y_multi: np.ndarray) -> np.ndarray:
    cum = np.cumsum(y_multi, axis=0)
    blocks = []
    for w in WINDOWS:
        rolled = np.zeros_like(cum, dtype=np.float32)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        blocks.append(rolled / float(w))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def gap_matrix(rows: np.ndarray) -> np.ndarray:
    n = rows.shape[0]
    gap = np.zeros((n, N_MAX), dtype=np.float32)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i, row in enumerate(rows):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in row:
            last_seen[v - 1] = i
    return gap


def make_sequences(features: np.ndarray, targets: np.ndarray, look_back: int):
    X, Y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back:i])
        Y.append(targets[i])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def topk_from_scores(scores_1d: np.ndarray, k: int = K) -> np.ndarray:
    scores = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -scores))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d: np.ndarray, y_true: np.ndarray) -> float:
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick: np.ndarray) -> str:
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


# =========================
# Feature engineering
# =========================
Y_full = draws_to_multihot(draws)
rolling_raw = rolling_features(Y_full)
gap_raw = gap_matrix(draws)

sum_col = draws.sum(axis=1, keepdims=True).astype(np.float32)
odd_col = (draws % 2 == 1).sum(axis=1, keepdims=True).astype(np.float32)
low_col = (draws <= 19).sum(axis=1, keepdims=True).astype(np.float32)
range_col = (draws.max(axis=1, keepdims=True) - draws.min(axis=1, keepdims=True)).astype(np.float32)
avg_gap_col = gap_raw.mean(axis=1, keepdims=True).astype(np.float32)
stats_raw = np.concatenate([sum_col, odd_col, low_col, range_col, avg_gap_col], axis=1)

step_features_raw = np.concatenate([Y_full, rolling_raw, gap_raw, stats_raw], axis=1).astype(np.float32)

START = max(LOOK_BACK, max(WINDOWS))
feature_scaler = StandardScaler()
step_features_scaled = step_features_raw.copy()
step_features_scaled[START:] = feature_scaler.fit_transform(step_features_raw[START:]).astype(np.float32)
step_features_scaled[:START] = feature_scaler.transform(step_features_raw[:START]).astype(np.float32)

X_all, Y_all = make_sequences(step_features_scaled, Y_full, LOOK_BACK)
X_all = X_all[START - LOOK_BACK:]
Y_all = Y_all[START - LOOK_BACK:]

n_total = X_all.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

X_train_full, Y_train_full = X_all[:n_train], Y_all[:n_train]
X_tr, Y_tr = X_train_full[:-VAL_N], Y_train_full[:-VAL_N]
X_val, Y_val = X_train_full[-VAL_N:], Y_train_full[-VAL_N:]
X_back, Y_back = X_all[n_train:], Y_all[n_train:]
X_next = step_features_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, step_features_scaled.shape[1]).astype(np.float32)

print(f"Feature dim: {X_all.shape[-1]}")
print(f"Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Back-test: {X_back.shape[0]}")
print()


class LotoTFT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        # Variable selection: uči težinu svakog feature-a u svakom vremenskom koraku.
        self.temporal_selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_MAX),
        )

    def forward(self, x):
        weights = self.temporal_selector(x)
        selected = x * weights
        projected = self.input_projection(selected)
        lstm_out, _ = self.lstm_encoder(projected)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        fused = self.norm1(lstm_out + attn_out)
        last = self.norm2(fused[:, -1, :])
        logits = self.head(last)
        return logits, weights, attn_weights


def make_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict_scores(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(X[start:start + BATCH])
            logits, _, _ = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def evaluate(model: nn.Module, X: np.ndarray, Y: np.ndarray):
    scores = predict_scores(model, X)
    h = avg_hits(scores, Y)
    a = safe_auc(Y, scores)
    l = safe_lrap(Y, scores)
    return scores, h, a, l


pos_weight_value = (N_MAX - K) / K
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_MAX,), pos_weight_value, dtype=torch.float32))

model = LotoTFT(input_dim=X_all.shape[-1])
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=50,
)

train_loader = make_loader(X_tr, Y_tr, BATCH, shuffle=False)
best_state = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
best_epoch = 0

print("Treniranje 1_TFT_loto_v2 ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    seen = 0
    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += float(loss.detach().cpu()) * xb.size(0)
        seen += xb.size(0)

    train_loss /= max(seen, 1)
    model.eval()
    with torch.no_grad():
        val_logits, _, _ = model(torch.from_numpy(X_val))
        val_loss = float(criterion(val_logits, torch.from_numpy(Y_val)).detach().cpu())
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
        print(f"epoch {epoch:4d}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  best_epoch={best_epoch}")

final_state = copy.deepcopy(model.state_dict())

print()
print(f"✅ 1_TFT_loto_v2 trening završen. best_epoch={best_epoch}, best_val_loss={best_val_loss:.5f}")
print()


# =========================
# BEST / FINAL evaluacija i predikcija sledećeg kola
# =========================
model.load_state_dict(best_state)
scores_best, h_best, auc_best, lrap_best = evaluate(model, X_back, Y_back)
next_best = predict_scores(model, X_next)[0]
pick_best = topk_from_scores(next_best)

model.load_state_dict(final_state)
scores_final, h_final, auc_final, lrap_final = evaluate(model, X_back, Y_back)
next_final = predict_scores(model, X_next)[0]
pick_final = topk_from_scores(next_final)

ensemble_scores = (scores_best + scores_final) / 2.0
h_ens = avg_hits(ensemble_scores, Y_back)
auc_ens = safe_auc(Y_back, ensemble_scores)
lrap_ens = safe_lrap(Y_back, ensemble_scores)
pick_ens = topk_from_scores((next_best + next_final) / 2.0)

for name, pick in [("TFT_best", pick_best), ("TFT_final", pick_final), ("TFT_ensemble", pick_ens)]:
    assert len(set(pick.tolist())) == K, f"{name} nema 7 jedinstvenih brojeva"
    assert pick.min() >= N_MIN and pick.max() <= N_MAX, f"{name} van opsega"
    assert list(pick) == sorted(pick.tolist()), f"{name} nije sortiran"

print("Predikcija sledeće Loto 7/39 kombinacije:")
print(f"TFT_best     -> {pick_best.tolist()}  ({describe(pick_best)})")
print(f"TFT_final    -> {pick_final.tolist()}  ({describe(pick_final)})")
print(f"TFT_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})")
print()

print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<12} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'TFT_best':<12} {h_best:>8.3f} {100*h_best/K:>6.1f}% {auc_best:>7.3f} {lrap_best:>7.3f}")
print(f"{'TFT_final':<12} {h_final:>8.3f} {100*h_final/K:>6.1f}% {auc_final:>7.3f} {lrap_final:>7.3f}")
print(f"{'TFT_ensemble':<12} {h_ens:>8.3f} {100*h_ens/K:>6.1f}% {auc_ens:>7.3f} {lrap_ens:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


elapsed = time.time() - T0
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N}, epochs={EPOCHS}) ---\n")
    f.write(f"TFT_best     -> {pick_best.tolist()}  ({describe(pick_best)})\n")
    f.write(f"TFT_final    -> {pick_final.tolist()}  ({describe(pick_final)})\n")
    f.write(f"TFT_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})\n")
    f.write(
        f"back-test: BEST hits/7={h_best:.3f}, AUC={auc_best:.3f}, LRAP={lrap_best:.3f}; "
        f"FINAL hits/7={h_final:.3f}, AUC={auc_final:.3f}, LRAP={lrap_final:.3f}; "
        f"ENSEMBLE hits/7={h_ens:.3f}, AUC={auc_ens:.3f}, LRAP={lrap_ens:.3f}; "
        f"baseline={7*7/39:.3f}\n"
    )
    f.write(f"elapsed={elapsed:.1f}s\n")

print(f"Snimljeno u: {OUT_TXT}")
print()
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()


"""
START 1_TFT_loto_v2.py 2026-05-24 22:47:37.331691

CSV učitan: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

Feature dim: 200
Train: 4220, Val: 200, Back-test: 100

Treniranje 1_TFT_loto_v2 ...
epoch    1/462  train_loss=1.13981  val_loss=1.13737  best_epoch=1
epoch   50/462  train_loss=0.67239  val_loss=1.89624  best_epoch=1
epoch  100/462  train_loss=0.45352  val_loss=2.65908  best_epoch=1
epoch  150/462  train_loss=0.37352  val_loss=3.08283  best_epoch=1
epoch  200/462  train_loss=0.34313  val_loss=3.29465  best_epoch=1
epoch  250/462  train_loss=0.33063  val_loss=3.40113  best_epoch=1
epoch  300/462  train_loss=0.32079  val_loss=3.49317  best_epoch=1
epoch  350/462  train_loss=0.31705  val_loss=3.51575  best_epoch=1
epoch  400/462  train_loss=0.31721  val_loss=3.53077  best_epoch=1
epoch  450/462  train_loss=0.31574  val_loss=3.53911  best_epoch=1
epoch  462/462  train_loss=0.31612  val_loss=3.54052  best_epoch=1

✅ 1_TFT_loto_v2 trening završen. best_epoch=1, best_val_loss=1.13737

Predikcija sledeće Loto 7/39 kombinacije:
TFT_best     -> [2, 7, 9, 25, 29, 34, 37]  (suma=143, neparnih=5/7, niskih(<=19)=3/7, raspon=35)
TFT_final    -> [4, 20, 24, 26, 28, 30, 38]  (suma=170, neparnih=0/7, niskih(<=19)=1/7, raspon=34)
TFT_ensemble -> [4, 20, 24, 26, 28, 30, 38]  (suma=170, neparnih=0/7, niskih(<=19)=1/7, raspon=34)

Back-test (poslednjih 100 izvlačenja):
model          hits/7    hit%     AUC    LRAP
TFT_best        1.260   18.0%   0.501   0.251
TFT_final       1.330   19.0%   0.510   0.263
TFT_ensemble    1.330   19.0%   0.511   0.263
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /1_TFT_loto_v2_predikcija.txt

STOP 2026-05-24 22:54:44.776999
Ukupno vreme: 0:07:07  (427.4 s)
"""
