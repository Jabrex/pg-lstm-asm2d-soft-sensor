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
from physics_consistency import compute_ode_residual_norm

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(ROOT, "synthetic_asm2d_physics.csv")
HP_PATH   = os.path.join(ROOT, "optuna_best_params.json")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(ROOT, "paper_artifacts")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
SEEDS        = [42, 123, 456]
EPOCHS       = 80
PATIENCE     = 15
SEQ_LEN = 24
TEST_FRAC    = 0.30

with open(HP_PATH) as f: HP = json.load(f)


def load_data():
    df = pd.read_csv(CSV_PATH)
    feat = ['KOI_giris', 'NH4_giris', 'PO4_giris', 'DO']
    tgt  = ['KOI_cikis', 'NH4_cikis', 'PO4_cikis']
    return df[feat + tgt].values.astype(np.float32)


def make_seqs(data, seq_len=SEQ_LEN):
    X = np.array([data[i:i+seq_len, :4] for i in range(len(data) - seq_len)])
    y = np.array([data[i:i+seq_len, 4:7] for i in range(len(data) - seq_len)])
    return X, y


def train_model(model_type, train_data, val_data, scaler, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    X_tr, y_tr = make_seqs(train_data)
    X_v, y_v   = make_seqs(val_data)
    raw_tr = X_tr.copy(); raw_v = X_v.copy()
    train_ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr),
                                         torch.FloatTensor(raw_tr)),
        batch_size=HP['batch_size'], shuffle=True, drop_last=True)
    val_ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_v), torch.FloatTensor(y_v),
                                         torch.FloatTensor(raw_v)),
        batch_size=HP['batch_size'], shuffle=False)

    model = ASM2d_PG_LSTM(input_dim=4, hidden_dim=HP['hidden_dim'],
        num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim']).to(DEVICE)
    NOISE_TRAIN = HP.get('noise_std', 0.05)

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

    best_loss, patience_cnt, best_state = float('inf'), 0, None
    for epoch in range(EPOCHS):
        model.train()
        for X, y, raw in train_ld:
            X, y, raw = X.to(DEVICE), y.to(DEVICE), raw.to(DEVICE)
            if NOISE_TRAIN > 0:
                scale = torch.rand(1, device=DEVICE).item() * NOISE_TRAIN
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


def eval_with_noise(model, test_data, scaler, noise_sigma):
    X_te, y_te = make_seqs(test_data)
    raw_te = X_te.copy()
    rng = np.random.default_rng(0)
    X_te_noisy = X_te + rng.normal(0, noise_sigma, X_te.shape).astype(np.float32)
    X_te_noisy = np.clip(X_te_noisy, 0.0, 1.0)
    val_ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_te_noisy), torch.FloatTensor(y_te),
                                         torch.FloatTensor(raw_te)),
        batch_size=HP['batch_size'], shuffle=False)
    model.eval()
    all_p, all_t, all_raw = [], [], []
    with torch.no_grad():
        for X, y, raw in val_ld:
            p = model(X.to(DEVICE))
            all_p.append(p[:, -1, :].cpu().numpy())
            all_t.append(y[:, -1, :].cpu().numpy())
            all_raw.append(raw[:, -1, :].numpy())
    pn, tn = np.concatenate(all_p), np.concatenate(all_t)
    raw_last = np.concatenate(all_raw)
    n = scaler.n_features_in_
    dp = np.zeros((len(pn), n)); dp[:, 4:7] = pn
    dtg = np.zeros((len(tn), n)); dtg[:, 4:7] = tn
    pr = scaler.inverse_transform(dp)[:, 4:7]
    tr = scaler.inverse_transform(dtg)[:, 4:7]
    metrics = {}
    for i, name in enumerate(['KOI', 'NH4', 'PO4']):
        metrics[f'{name}_R2']   = r2_score(tr[:, i], pr[:, i])
        metrics[f'{name}_RMSE'] = np.sqrt(mean_squared_error(tr[:, i], pr[:, i]))
    try:
        metrics['ODE_res_mean'] = compute_ode_residual_norm(pr, raw_last, dt=1.0, reduce='mean_l2')
    except Exception:
        metrics['ODE_res_mean'] = float('nan')
    return metrics


def run():
    print(f"Noise Robustness | Cihaz: {DEVICE}")
    full = load_data()
    n_test = int(len(full) * TEST_FRAC)
    train_pool = full[:len(full) - n_test]
    test_data  = full[len(full) - n_test:]
    sc = MinMaxScaler(); sc.fit(train_pool)
    train_norm = sc.transform(train_pool)
    test_norm  = sc.transform(test_data)
    val_norm   = train_norm[-len(test_norm)//2:]

    records = []
    for model_type in ["PG_LSTM", "Vanilla_LSTM"]:
        for seed in SEEDS:
            print(f"\n  Eğitim: [{model_type} seed={seed}]")
            model = train_model(model_type, train_norm[:-len(val_norm)], val_norm, sc, seed)
            for sigma in NOISE_LEVELS:
                m = eval_with_noise(model, test_norm, sc, sigma)
                row = {'Model': model_type, 'Seed': seed, 'NoiseSigma': sigma, **m}
                records.append(row)
                print(f"    σ={sigma:.2f}: KOI_R2={m['KOI_R2']:.4f}  NH4_R2={m['NH4_R2']:.4f}  ODE={m['ODE_res_mean']:.2f}")

    df = pd.DataFrame(records)
    csv_out = os.path.join(OUT_DIR, "noise_robustness_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n→ {csv_out}")

    summary = df.groupby(['Model', 'NoiseSigma']).agg(
        koi_r2_mean=('KOI_R2', 'mean'), koi_r2_std=('KOI_R2', 'std'),
        nh4_r2_mean=('NH4_R2', 'mean'), nh4_r2_std=('NH4_R2', 'std'),
        ode_mean=('ODE_res_mean', 'mean'), ode_std=('ODE_res_mean', 'std')
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_display = {"PG_LSTM": "PG-LSTM", "Vanilla_LSTM": "Vanilla LSTM"}
    for ax, (col_m, col_s, ylabel) in zip(
        axes,
        [('koi_r2_mean', 'koi_r2_std', 'COD R²'),
         ('nh4_r2_mean', 'nh4_r2_std', 'NH4 R²'),
         ('ode_mean',    'ode_std',    'ODE Residual (mean)')]):
        for mt, color in [("PG_LSTM", "#E63946"), ("Vanilla_LSTM", "#457B9D")]:
            s = summary[summary.Model == mt].sort_values('NoiseSigma')
            ax.errorbar(s.NoiseSigma, s[col_m], yerr=s[col_s],
                         label=model_display[mt], marker='o', linewidth=2, capsize=4, color=color)
        ax.set_xlabel('Test-time Input Noise σ'); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    fig.suptitle('Path B Domain of Utility — Noise Robustness (Test-time)')
    plt.tight_layout()
    fig_out = os.path.join(PAPER_DIR, "fig_noise_curve.png")
    fig.savefig(fig_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"→ {fig_out}")


if __name__ == "__main__":
    run()
