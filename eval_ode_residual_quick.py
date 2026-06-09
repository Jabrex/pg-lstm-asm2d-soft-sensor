import os, sys, torch, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch.nn as nn

from asm2d_pg_lstm import ASM2d_PG_LSTM
from baselines.gru_model import ASM2d_GRU
from physics_consistency import compute_ode_residual_norm

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = os.path.join(os.path.dirname(__file__), "synthetic_asm2d_physics.csv")
SEQ_LEN = 24
TRAIN_RATIO = 0.8

print(f"Device: {DEVICE}")

df = pd.read_csv(CSV_PATH)
feat = ['KOI_giris', 'NH4_giris', 'PO4_giris', 'DO']
tgt  = ['KOI_cikis', 'NH4_cikis', 'PO4_cikis']
data = df[feat + tgt].values.astype(np.float32)

split_idx = int(len(data) * TRAIN_RATIO)
scaler = MinMaxScaler()
scaler.fit(data[:split_idx])
scaled = scaler.transform(data)

X_seqs, y_seqs, raw_seqs = [], [], []
for i in range(len(scaled) - SEQ_LEN):
    X_seqs.append(scaled[i:i+SEQ_LEN, :4])
    y_seqs.append(scaled[i:i+SEQ_LEN, 4:7])
    raw_seqs.append(data[i:i+SEQ_LEN, :4])

X = np.array(X_seqs); y = np.array(y_seqs); raw = np.array(raw_seqs)
split = int(len(X) * TRAIN_RATIO)
gap   = SEQ_LEN
X_test = X[split + gap:]
y_test = y[split + gap:]
raw_test = raw[split + gap:]

raw_last = raw_test[:, -1, :]

def denorm(pn):
    dummy = np.zeros((len(pn), 7))
    dummy[:, 4:7] = pn
    return scaler.inverse_transform(dummy)[:, 4:7]

def eval_model(model, name):
    model.eval()
    model.to(DEVICE)
    X_t = torch.FloatTensor(X_test).to(DEVICE)
    with torch.no_grad():
        preds_norm = model(X_t)[:, -1, :].cpu().numpy()
    pr = denorm(preds_norm)
    tr = denorm(y_test[:, -1, :])

    metrics = {}
    for i, n in enumerate(['KOI', 'NH4', 'PO4']):
        metrics[f'{n}_R2']   = r2_score(tr[:, i], pr[:, i])

    try:
        per_state = compute_ode_residual_norm(pr, raw_last, dt=1.0, reduce='per_state')
        metrics['ODE_SS']   = per_state['SS']
        metrics['ODE_SNH']  = per_state['SNH']
        metrics['ODE_SPO4'] = per_state['SPO4']
        metrics['ODE_mean'] = compute_ode_residual_norm(pr, raw_last, dt=1.0, reduce='mean_l2')
    except Exception as e:
        print(f"  [WARN] ODE calc failed for {name}: {e}")
        metrics['ODE_mean'] = float('nan')

    return pr, metrics

import json
with open("optuna_best_params.json") as f: HP = json.load(f)

pg = ASM2d_PG_LSTM(input_dim=4, hidden_dim=HP['hidden_dim'],
    num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim'])
ckpt = torch.load("asm2d_pg_lstm_model.pth", map_location='cpu', weights_only=False)
pg.load_state_dict(ckpt['model_state_dict'])
_, m_pg = eval_model(pg, "PG_LSTM")

vlstm = ASM2d_PG_LSTM(input_dim=4, hidden_dim=HP['hidden_dim'],
    num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim'])
vlstm.load_state_dict(torch.load("baselines/vanilla_lstm_model.pth",
    map_location='cpu', weights_only=False))
_, m_vlstm = eval_model(vlstm, "Vanilla_LSTM")

gru = ASM2d_GRU(input_dim=4, hidden_dim=HP['hidden_dim'],
    num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim'])
gru.load_state_dict(torch.load("baselines/gru_model.pth",
    map_location='cpu', weights_only=False))
_, m_gru = eval_model(gru, "GRU")

tr_data = denorm(y_test[:, -1, :])
pr_persist = np.roll(tr_data, 1, axis=0); pr_persist[0] = tr_data[0]
m_persist = {}
for i, n in enumerate(['KOI', 'NH4', 'PO4']):
    m_persist[f'{n}_R2'] = r2_score(tr_data[1:, i], pr_persist[1:, i])
try:
    m_persist['ODE_mean'] = compute_ode_residual_norm(pr_persist, raw_last, dt=1.0, reduce='mean_l2')
    per_s = compute_ode_residual_norm(pr_persist, raw_last, dt=1.0, reduce='per_state')
    m_persist['ODE_SS'] = per_s['SS']; m_persist['ODE_SNH'] = per_s['SNH']
    m_persist['ODE_SPO4'] = per_s['SPO4']
except Exception as e:
    print(f"  [WARN] Persistence ODE failed: {e}"); m_persist['ODE_mean'] = float('nan')

print("\n" + "="*70)
print("CORRECTED ODE RESIDUAL (CSTR dilution dahil, D=0.1667 /h)")
print("="*70)
for name, m in [("PG_LSTM", m_pg), ("Vanilla_LSTM", m_vlstm),
                ("GRU", m_gru), ("Persistence", m_persist)]:
    print(f"\n{name}:")
    print(f"  R²: KOI={m['KOI_R2']:.4f}  NH4={m['NH4_R2']:.4f}  PO4={m['PO4_R2']:.4f}")
    if 'ODE_SS' in m:
        print(f"  ODE per-state: SS={m['ODE_SS']:.1f}  SNH={m['ODE_SNH']:.2f}  SPO4={m['ODE_SPO4']:.3f}")
    print(f"  ODE mean_l2: {m.get('ODE_mean', float('nan')):.2f}")

print("\n--- PG/VLSTM ODE ratio ---")
ratio_pv = m_vlstm.get('ODE_mean', 0) / max(m_pg.get('ODE_mean', 1), 1e-9)
ratio_pg = m_gru.get('ODE_mean', 0) / max(m_pg.get('ODE_mean', 1), 1e-9)
ratio_pp = m_persist.get('ODE_mean', 0) / max(m_pg.get('ODE_mean', 1), 1e-9)
print(f"  VLSTM/PG = {ratio_pv:.2f}×")
print(f"  GRU/PG   = {ratio_pg:.2f}×")
print(f"  Persistence/PG = {ratio_pp:.2f}×")
print("\n[Faz 3 karar: ≥1.5× → Senaryo A; <1.5× → Senaryo B]")
