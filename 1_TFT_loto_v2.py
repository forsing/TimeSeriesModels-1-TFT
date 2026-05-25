#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Hibridne arhitekture za predikciju koje kombinuju deep learning i klasične time-series modele.

1. Temporal Fusion Transformers (TFT) for Multivariate Forecasting


Loto 7/39 (loto7hh_4620_k41.csv):
 • Polazna klasa TemporalFusionTransformer (variable selection + LSTM + multi-head attention). 
   TemporalFusionTransformer (static_embedding, temporal_selector, lstm_encoder, attention)
   Jedna sigmoid glava sa 39 izlaza (po jedan za svaki broj 1..39).
 • Static covariate: jedan placeholder token (n_static_categories=1) 
   (embedding + concat sa temporal feature-ima).
 • Ulaz: poslednjih LOOK_BACK kola, feature-i:
     - 39 multi-hot brojeva
     - rolling frekvencije 20/50/100 za brojeve 1..39
     - gap za brojeve 1..39
     - 4 statistike kola (suma, neparni, niski, raspon)
 • Loss: BCEWithLogits + pos_weight = (39-7)/7 ≈ 4.57
 • Vremenski split train/val/back-test, bez shuffle.
 • BEST / FINAL / ENSEMBLE + back-test poslednjih 100: hits/7, AUC, LRAP.
 • SEED=39, single-thread, PyTorch deterministic.
 • Snima u 1_TFT_loto_v2_predikcija.txt.
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


N_MIN, N_MAX = 1, 39
K = 7


class TemporalFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Static covariate encoder: transforms categorical metadata
        self.static_embedding = nn.Embedding(
            num_embeddings=config['n_static_categories'],
            embedding_dim=config['static_dim']
        )
        
        # Variable selection networks for efficient feature pruning
        fused_dim = config['input_dim'] + config['static_dim']
        self.temporal_selector = nn.Sequential(
            nn.Linear(fused_dim, config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config['hidden_dim'], fused_dim),
            nn.Sigmoid()  # Produces feature importance weights
        )
        
        # LSTM encoder for local temporal patterns
        self.lstm_encoder = nn.LSTM(
            input_size=fused_dim,
            hidden_size=config['hidden_dim'],
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Multi-head attention for long-term dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=config['hidden_dim'],
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Loto izlaz: 39 sigmoid logita (multi-label po broju), umesto kvantilnih glava
        self.output_head = nn.Linear(config['hidden_dim'], N_MAX)
        
        self.config = config
        
    def forward(self, x, static_covariates):
        """
        Args:
            x: Temporal features [batch, history_len, features]
            static_covariates: Static metadata [batch, 1]
        """
        # Encode static covariates (e.g., store ID, region)
        static_emb = self.static_embedding(static_covariates).squeeze(1).unsqueeze(1)  # [batch, 1, static_dim]
        static_emb = static_emb.expand(-1, x.size(1), -1)
        
        # Fuse static and temporal features
        fused_input = torch.cat([x, static_emb], dim=-1)
        
        # Variable selection: prune irrelevant features dynamically
        importance_weights = self.temporal_selector(fused_input)
        selected_features = fused_input * importance_weights
        
        # LSTM processes selected features
        lstm_out, _ = self.lstm_encoder(selected_features)
        
        # Self-attention over LSTM outputs
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 39 logita iz poslednjeg vremenskog koraka (multi-label izlaz)
        last = attn_out[:, -1, :]
        logits = self.output_head(last)
        
        return logits, importance_weights, attn_weights


# =========================
# Učitavanje Loto 7/39 CSV-a
# =========================
CSV_PATH = "/loto7hh_4620_k41.csv"
OUT_TXT = Path("/1_TFT_loto_v2_predikcija.txt")

LOOK_BACK = 10
WINDOWS = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 100
BATCH = 64
LR = 1e-3
HIDDEN_DIM = 128
STATIC_DIM = 16

T0 = time.time()
print()
print("START 1_TFT_loto_v2", datetime.today())
print()

df = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV: {CSV_PATH}")
print(f"Broj izvlačenja: {N}, brojeva po kolu: {K}")
print()


# =========================
# Feature engineering
# =========================
def draws_to_multihot(rows):
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def rolling_features(y_multi):
    cum = np.cumsum(y_multi, axis=0)
    blocks = []
    for w in WINDOWS:
        rolled = np.zeros_like(cum, dtype=np.float32)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        blocks.append(rolled / float(w))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def gap_matrix(rows):
    n = rows.shape[0]
    gap = np.zeros((n, N_MAX), dtype=np.float32)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i, row in enumerate(rows):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in row:
            last_seen[v - 1] = i
    return gap


def make_sequences(features, targets, look_back):
    X, Y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back:i])
        Y.append(targets[i])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d, y_true):
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true, scores):
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick):
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)
rolling_raw = rolling_features(Y_full)
gap_raw = gap_matrix(draws)

sum_col = draws.sum(axis=1, keepdims=True).astype(np.float32)
odd_col = (draws % 2 == 1).sum(axis=1, keepdims=True).astype(np.float32)
low_col = (draws <= 19).sum(axis=1, keepdims=True).astype(np.float32)
range_col = (draws.max(axis=1, keepdims=True) - draws.min(axis=1, keepdims=True)).astype(np.float32)
stats_raw = np.concatenate([sum_col, odd_col, low_col, range_col], axis=1)

step_features_raw = np.concatenate([Y_full, rolling_raw, gap_raw, stats_raw], axis=1).astype(np.float32)

START = max(LOOK_BACK, max(WINDOWS))
feature_scaler = StandardScaler()
step_features = step_features_raw.copy()
step_features[START:] = feature_scaler.fit_transform(step_features_raw[START:]).astype(np.float32)
step_features[:START] = feature_scaler.transform(step_features_raw[:START]).astype(np.float32)

X_seq, Y_seq = make_sequences(step_features, Y_full, LOOK_BACK)
X_seq = X_seq[START - LOOK_BACK:]
Y_seq = Y_seq[START - LOOK_BACK:]

n_total = X_seq.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

X_tr_seq, Y_tr_seq = X_seq[:n_train - VAL_N], Y_seq[:n_train - VAL_N]
X_val_seq, Y_val_seq = X_seq[n_train - VAL_N:n_train], Y_seq[n_train - VAL_N:n_train]
X_back_seq, Y_back = X_seq[n_train:], Y_seq[n_train:]
X_next_seq = step_features[-LOOK_BACK:].reshape(1, LOOK_BACK, step_features.shape[1]).astype(np.float32)

INPUT_DIM = X_seq.shape[-1]
print(f"Feature dim: {INPUT_DIM}")
print(f"Train: {X_tr_seq.shape[0]}, Val: {X_val_seq.shape[0]}, Back-test: {X_back_seq.shape[0]}")
print()

# Static placeholder token (jedan jedinstveni "loto" kontekst)
static_tr = np.zeros((X_tr_seq.shape[0], 1), dtype=np.int64)
static_val = np.zeros((X_val_seq.shape[0], 1), dtype=np.int64)
static_back = np.zeros((X_back_seq.shape[0], 1), dtype=np.int64)
static_next = np.zeros((1, 1), dtype=np.int64)


# Konfiguracija TFT-a (kao u polaznom, samo prilagođeno na loto dimenzije)
config = {
    'n_static_categories': 1,
    'static_dim': STATIC_DIM,
    'input_dim': INPUT_DIM,
    'hidden_dim': HIDDEN_DIM,
}

model = TemporalFusionTransformer(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)

pos_weight_value = (N_MAX - K) / K
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_MAX,), pos_weight_value, dtype=torch.float32))


def make_loader(X, S, Y, batch_size, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(S), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


train_loader = make_loader(X_tr_seq, static_tr, Y_tr_seq, BATCH, shuffle=False)
val_X_t = torch.from_numpy(X_val_seq)
val_S_t = torch.from_numpy(static_val)
val_Y_t = torch.from_numpy(Y_val_seq)

best_state = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
best_epoch = 0

print("Treniranje TFT na loto podacima ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    seen = 0
    for xb, sb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(xb, sb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += float(loss.detach().cpu()) * xb.size(0)
        seen += xb.size(0)
    train_loss /= max(seen, 1)

    model.eval()
    with torch.no_grad():
        val_logits, _, _ = model(val_X_t, val_S_t)
        val_loss = float(criterion(val_logits, val_Y_t).detach().cpu())
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
        print(f"epoch {epoch:4d}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  best_epoch={best_epoch}")

final_state = copy.deepcopy(model.state_dict())
print()
print(f"✅ Trening završen. best_epoch={best_epoch}, best_val_loss={best_val_loss:.5f}")
print()


# =========================
# Predikcija sledećeg kola + back-test
# =========================
def predict_scores(model, X, S):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(X[start:start + BATCH])
            sb = torch.from_numpy(S[start:start + BATCH])
            logits, _, _ = model(xb, sb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def evaluate(model, X, S, Y):
    scores = predict_scores(model, X, S)
    return scores, avg_hits(scores, Y), safe_auc(Y, scores), safe_lrap(Y, scores)


model.load_state_dict(best_state)
scores_best, h_best, auc_best, lrap_best = evaluate(model, X_back_seq, static_back, Y_back)
next_best = predict_scores(model, X_next_seq, static_next)[0]
pick_best = topk_from_scores(next_best)

model.load_state_dict(final_state)
scores_final, h_final, auc_final, lrap_final = evaluate(model, X_back_seq, static_back, Y_back)
next_final = predict_scores(model, X_next_seq, static_next)[0]
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
print(f"{'model':<16} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'TFT_best':<16} {h_best:>8.3f} {100*h_best/K:>6.1f}% {auc_best:>7.3f} {lrap_best:>7.3f}")
print(f"{'TFT_final':<16} {h_final:>8.3f} {100*h_final/K:>6.1f}% {auc_final:>7.3f} {lrap_final:>7.3f}")
print(f"{'TFT_ensemble':<16} {h_ens:>8.3f} {100*h_ens/K:>6.1f}% {auc_ens:>7.3f} {lrap_ens:>7.3f}")
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
print()
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()



"""
START 1_TFT_loto_v2 2026-05-25 08:11:31.264252

CSV: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

Feature dim: 199
Train: 4220, Val: 200, Back-test: 100 
(prvih 100 se otpisuje)

Treniranje TFT na loto podacima ...
epoch    1/100  train_loss=1.13783  val_loss=1.13710  best_epoch=1
epoch   50/100  train_loss=0.40826  val_loss=3.46261  best_epoch=1
epoch  100/100  train_loss=0.14622  val_loss=7.08012  best_epoch=1

✅ Trening završen. best_epoch=1, best_val_loss=1.13710

Predikcija sledeće Loto 7/39 kombinacije:
TFT_best     -> [8, x, 11, y, 26, z, 37]  (suma=144, neparnih=4/7, niskih(<=19)=3/7, raspon=29)
TFT_final    -> [3, x, 7, y, 20, z, 27]  (suma=100, neparnih=4/7, niskih(<=19)=4/7, raspon=24)
TFT_ensemble -> [3, x, 7, y, 20, z, 27]  (suma=100, neparnih=4/7, niskih(<=19)=4/7, raspon=24)

Back-test (poslednjih 100 izvlačenja):
model              hits/7    hit%     AUC    LRAP
TFT_best            1.230   17.6%   0.502   0.245
TFT_final           1.130   16.1%   0.504   0.248
TFT_ensemble        1.150   16.4%   0.503   0.247
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /1_TFT_loto_v2_predikcija.txt

STOP 2026-05-25 08:13:04.744353

Ukupno vreme: 0:01:33  (93.5 s)
"""
