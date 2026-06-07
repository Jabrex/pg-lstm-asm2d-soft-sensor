import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from asm2d_pg_lstm import ASM2d_PG_LSTM, PGLossFunction

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(ROOT, "synthetic_asm2d_physics.csv")
HP_PATH   = os.path.join(ROOT, "optuna_best_params.json")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(ROOT, "paper_artifacts")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HORIZONS  = [1, 6, 24, 72, 168]
SEEDS     = [42, 123, 456]
EPOCHS    = 80
PATIENCE  = 15
SEQ_LEN   = 48

with open(HP_PATH) as f: HP = json.load(f)


def load_data_split(train_frac=0.7):
    df = pd.read_csv(CSV_PATH)
    feat = ['KOI_giris', 'NH4_giris', 'PO4_giris', 'DO']
    tgt  = ['KOI_cikis', 'NH4_cikis', 'PO4_cikis']
    data = df[feat + tgt].values.astype(np.float32)
    n_train = int(len(data) * train_frac)
    train_raw = data[:n_train]
    test_raw  = data[n_train:]
    sc = MinMaxScaler()
    sc.fit(train_raw)
    train = sc.transform(train_raw)
    test  = sc.transform(test_raw)
    return train, test, sc


def make_seqs(data, seq_len=SEQ_LEN):
    X = np.array([data[i:i+seq_len, :4] for i in range(len(data) - seq_len)])
    y = np.array([data[i:i+seq_len, 4:7] for i in range(len(data) - seq_len)])
    return X, y


def train_model(model, X_tr, y_tr, X_val, y_val, model_type, scaler, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = model.to(DEVICE)
    NOISE_STD = HP.get('noise_std', 0.05)

    if model_type == "PG_LSTM":
        loss_fn_phys = PGLossFunction(
            lambda_physics=0.05, lambda_variance=0.102, lambda_boundary=0.04,
            out_range=scaler.data_range_[4:7], out_min=scaler.data_min_[4:7])
        loss_fn = lambda p, y, r: loss_fn_phys(p, y, r)[0]
    else:
        mse = nn.MSELoss()
        loss_fn = lambda p, y, r: mse(p, y)

    opt = optim.AdamW(model.parameters(), lr=HP['lr'], weight_decay=HP['weight_decay'])
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

    Xt = torch.FloatTensor(X_tr); yt = torch.FloatTensor(y_tr)
    Xv = torch.FloatTensor(X_val); yv = torch.FloatTensor(y_val)
    raw_t = Xt.clone(); raw_v = Xv.clone()
    train_ds = torch.utils.data.TensorDataset(Xt, yt, raw_t)
    val_ds   = torch.utils.data.TensorDataset(Xv, yv, raw_v)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=HP['batch_size'], shuffle=True, drop_last=True)
    val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=HP['batch_size'], shuffle=False)

    best_loss, patience_cnt, best_state = float('inf'), 0, None
    for epoch in range(EPOCHS):
        model.train()
        for X, y, raw in train_ld:
            X, y, raw = X.to(DEVICE), y.to(DEVICE), raw.to(DEVICE)
            if NOISE_STD > 0:
                scale = torch.rand(1, device=DEVICE).item() * NOISE_STD
                X = torch.clamp(X + scale * torch.randn_like(X), 0.0, 1.0)
            opt.zero_grad()
            loss_fn(model(X), y, raw).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        vl = 0
        with torch.no_grad():
            for X, y, raw in val_ld:
                X, y, raw = X.to(DEVICE), y.to(DEVICE), raw.to(DEVICE)
                vl += loss_fn(model(X), y, raw).item()
        vl /= max(1, len(val_ld))
        if vl < best_loss:
            best_loss, patience_cnt = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def autoregressive_forecast(model, init_seq, future_inputs, horizon):
    model.eval()
    seq = torch.FloatTensor(init_seq).unsqueeze(0).to(DEVICE)
    preds = []
    with torch.no_grad():
        for h in range(horizon):
            out = model(seq)[:, -1, :]
            preds.append(out.cpu().numpy().squeeze())
            next_in = torch.FloatTensor(future_inputs[h:h+1]).unsqueeze(0).to(DEVICE)
            seq = torch.cat([seq[:, 1:, :], next_in], dim=1)
    return np.array(preds)


def evaluate_horizon(model, test_seqs_X, test_full_data, horizon, scaler, n_samples=200):
    n = len(test_full_data) - SEQ_LEN - horizon
    if n < n_samples: n_samples = n
    if n_samples <= 0:
        return float('nan'), float('nan')
    indices = np.linspace(0, n - 1, n_samples).astype(int)
    all_preds, all_targets = [], []
    for idx in indices:
        init_seq = test_full_data[idx:idx+SEQ_LEN, :4]
        future_in = test_full_data[idx+SEQ_LEN:idx+SEQ_LEN+horizon, :4]
        true_out  = test_full_data[idx+SEQ_LEN:idx+SEQ_LEN+horizon, 4:7]
        preds = autoregressive_forecast(model, init_seq, future_in, horizon)
        all_preds.append(preds)
        all_targets.append(true_out)
    P = np.concatenate(all_preds)
    T = np.concatenate(all_targets)
    n_feat = scaler.n_features_in_
    dp = np.zeros((len(P), n_feat)); dp[:, 4:7] = P
    dt = np.zeros((len(T), n_feat)); dt[:, 4:7] = T
    P_real = scaler.inverse_transform(dp)[:, 4:7]
    T_real = scaler.inverse_transform(dt)[:, 4:7]
    rmse = np.sqrt(mean_squared_error(T_real, P_real))
    r2   = r2_score(T_real.reshape(-1), P_real.reshape(-1))
    return rmse, r2


def run():
    print(f"Forecast Horizon Sweep | Cihaz: {DEVICE}")
    train_raw, test_raw, scaler = load_data_split(0.7)
    X_tr_seq, y_tr_seq = make_seqs(train_raw)
    X_val_seq, y_val_seq = make_seqs(test_raw[:len(test_raw)//2])

    records = []
    for model_type in ["PG_LSTM", "Vanilla_LSTM"]:
        for seed in SEEDS:
            print(f"\n  [{model_type} seed={seed}]")
            torch.manual_seed(seed)
            model = ASM2d_PG_LSTM(input_dim=4, hidden_dim=HP['hidden_dim'],
                num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim'])
            model = train_model(model, X_tr_seq, y_tr_seq, X_val_seq, y_val_seq,
                                 model_type, scaler, seed=seed)
            for h in HORIZONS:
                rmse, r2 = evaluate_horizon(model, X_val_seq, test_raw, h, scaler)
                print(f"    horizon={h:>3}h: RMSE={rmse:.4f}, R²={r2:.4f}")
                records.append({
                    'Model': model_type, 'Seed': seed, 'Horizon_h': h,
                    'RMSE': rmse, 'R2': r2,
                })

    df = pd.DataFrame(records)
    csv_out = os.path.join(OUT_DIR, "forecast_horizon_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n→ {csv_out}")

    summary = df.groupby(['Model', 'Horizon_h']).agg(
        rmse_mean=('RMSE', 'mean'), rmse_std=('RMSE', 'std'),
        r2_mean=('R2', 'mean'), r2_std=('R2', 'std')).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    model_display = {"PG_LSTM": "PG-LSTM", "Vanilla_LSTM": "Vanilla LSTM"}
    for model_type, color in [("PG_LSTM", "#E63946"), ("Vanilla_LSTM", "#457B9D")]:
        s = summary[summary.Model == model_type].sort_values('Horizon_h')
        ax1.errorbar(s.Horizon_h, s.rmse_mean, yerr=s.rmse_std,
                      label=model_display[model_type], marker='o', linewidth=2, capsize=4, color=color)
        ax2.errorbar(s.Horizon_h, s.r2_mean, yerr=s.r2_std,
                      label=model_display[model_type], marker='o', linewidth=2, capsize=4, color=color)
    ax1.set_xlabel('Forecast Horizon (h)'); ax1.set_ylabel('RMSE')
    ax1.set_title('Forecast Horizon vs RMSE'); ax1.set_xscale('log')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Forecast Horizon (h)'); ax2.set_ylabel('R²')
    ax2.set_title('Forecast Horizon vs R²'); ax2.set_xscale('log')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_out = os.path.join(PAPER_DIR, "fig_horizon_curve.png")
    fig.savefig(fig_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"→ {fig_out}")


if __name__ == "__main__":
    run()
