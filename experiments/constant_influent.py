"""
Yeni Deney 4 — Constant-Influent Steady-State Test
==================================================

Gemini önerisi: shuffled data yerine sabit influent → steady-state recovery.

Setup:
  Sabit influent vector = test set ortalaması (KOI_in_mean, NH4_in_mean, PO4_in_mean, DO_mean)
  Auto-regressive forward simulation 168 saat
  PG-LSTM ve Vanilla LSTM karşılaştır
  3 seed (3 farklı eğitilmiş model)

Beklenen (Path B):
  - PG-LSTM steady-state'a converge
  - Vanilla LSTM drift edebilir (mekanistik kısıt yok)
  - Final-state ODE residual: PG-LSTM < Vanilla

Çıktı:
  experiments/steady_state_results.csv
  paper_artifacts/fig_steady_state.png
"""
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
from sklearn.preprocessing import MinMaxScaler

from asm2d_pg_lstm import ASM2d_PG_LSTM, PGLossFunction
from physics_consistency import compute_ode_residual_norm

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(ROOT, "synthetic_asm2d_physics.csv")
HP_PATH   = os.path.join(ROOT, "optuna_best_params.json")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(ROOT, "paper_artifacts")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIM_HORIZON = 168       # 7 gün steady-state convergence
SEEDS       = [42, 123, 456]
EPOCHS      = 80
PATIENCE    = 15
SEQ_LEN     = 48
TEST_FRAC   = 0.30

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
    NOISE = HP.get('noise_std', 0.05)

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
            if NOISE > 0:
                scale = torch.rand(1, device=DEVICE).item() * NOISE
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


def steady_state_simulate(model, init_seq_norm, constant_input_norm, horizon, scaler):
    """
    init_seq_norm        : [seq_len, 4] normalize başlangıç
    constant_input_norm  : [4]          normalize sabit input vektörü
    horizon              : adım sayısı
    Returns: [horizon, 3] normalize predictions, then denormalize
    """
    model.eval()
    seq = torch.FloatTensor(init_seq_norm).unsqueeze(0).to(DEVICE)
    const_in = torch.FloatTensor(constant_input_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    preds_norm = []
    with torch.no_grad():
        for h in range(horizon):
            out = model(seq)[:, -1, :]
            preds_norm.append(out.cpu().numpy().squeeze())
            seq = torch.cat([seq[:, 1:, :], const_in], dim=1)
    preds_norm = np.array(preds_norm)
    n = scaler.n_features_in_
    dp = np.zeros((len(preds_norm), n)); dp[:, 4:7] = preds_norm
    return scaler.inverse_transform(dp)[:, 4:7]


def run():
    print(f"Constant-Influent Steady-State | Cihaz: {DEVICE}")
    full = load_data()
    n_test = int(len(full) * TEST_FRAC)
    train_pool = full[:len(full) - n_test]
    test_data  = full[len(full) - n_test:]
    sc = MinMaxScaler(); sc.fit(train_pool)
    train_norm = sc.transform(train_pool)
    test_norm  = sc.transform(test_data)
    val_split  = train_norm[-len(test_norm)//2:]

    # Sabit input — test set ortalaması (real units)
    mean_input_real = test_data[:, :4].mean(axis=0)
    mean_input_norm = sc.transform(np.hstack([mean_input_real, [0,0,0]]).reshape(1,-1))[0, :4]

    # Sabit DO için ASM2d steady-state ground truth (yaklaşık) hesabı yok — sadece convergence test
    # Initial seq: rastgele test penceresi
    init_seq_norm = test_norm[0:SEQ_LEN, :4]

    print(f"  Sabit input (real): COD={mean_input_real[0]:.2f}, NH4={mean_input_real[1]:.2f}, "
          f"PO4={mean_input_real[2]:.2f}, DO={mean_input_real[3]:.2f}")

    records = []
    trajectories = {}    # plotlama için
    for model_type in ["PG_LSTM", "Vanilla_LSTM"]:
        traj_per_seed = []
        for seed in SEEDS:
            print(f"\n  Eğitim: [{model_type} seed={seed}]")
            model = train_model(model_type, train_norm[:-len(val_split)], val_split, sc, seed)
            traj = steady_state_simulate(model, init_seq_norm, mean_input_norm, SIM_HORIZON, sc)
            traj_per_seed.append(traj)
            # Convergence metrik: son 24 saat std (steady → küçük)
            final_std = traj[-24:, :].std(axis=0)
            initial_val = traj[0]
            final_val   = traj[-1]
            # ODE residual son 24 saatte — sabit influent array [24, 4] gerekli
            raw_const = np.tile(mean_input_real[:4], (24, 1))  # [24, 4]: [KOI_in, NH4_in, PO4_in, DO]
            try:
                ode_final = compute_ode_residual_norm(traj[-24:], raw_const, dt=1.0, reduce='mean_l2')
            except Exception:
                ode_final = float('nan')
            records.append({
                'Model': model_type, 'Seed': seed,
                'Final_Std_COD': final_std[0], 'Final_Std_NH4': final_std[1], 'Final_Std_PO4': final_std[2],
                'Final_COD': final_val[0], 'Final_NH4': final_val[1], 'Final_PO4': final_val[2],
                'Initial_COD': initial_val[0], 'ODE_res_final': ode_final,
            })
            print(f"    final_std (COD/NH4/PO4) = {final_std[0]:.3f} / {final_std[1]:.3f} / {final_std[2]:.3f}")
        trajectories[model_type] = np.array(traj_per_seed)   # [n_seeds, horizon, 3]

    df = pd.DataFrame(records)
    csv_out = os.path.join(OUT_DIR, "steady_state_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n→ {csv_out}")

    # Plot — 3 outputs trajectory
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    out_names = ['COD Effluent (mg/L)', 'NH4 Effluent (mg-N/L)', 'PO4 Effluent (mg-P/L)']
    model_display = {"PG_LSTM": "PG-LSTM", "Vanilla_LSTM": "Vanilla LSTM"}
    t = np.arange(SIM_HORIZON)
    for i, ax in enumerate(axes):
        for mt, color in [("PG_LSTM", "#E63946"), ("Vanilla_LSTM", "#457B9D")]:
            traj_arr = trajectories[mt]   # [n_seeds, horizon, 3]
            mean_traj = traj_arr[:, :, i].mean(axis=0)
            std_traj  = traj_arr[:, :, i].std(axis=0)
            ax.plot(t, mean_traj, label=model_display[mt], linewidth=2, color=color)
            ax.fill_between(t, mean_traj - std_traj, mean_traj + std_traj, alpha=0.2, color=color)
            # Bireysel seed trajektorileri ince çizgi (Gemini önerisi — band misleading)
            for s_idx in range(traj_arr.shape[0]):
                ax.plot(t, traj_arr[s_idx, :, i], color=color, alpha=0.35, linewidth=0.8)
        ax.set_ylabel(out_names[i])
        # Sıfır referans çizgisi (sadece PO4 için anlamlı)
        if i == 2:
            ax.axhline(0.0, color='red', ls='-.', lw=0.8, alpha=0.5, label='Physical bound')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.axvline(24, color='gray', ls=':', alpha=0.5)
        ax.axvline(72, color='gray', ls=':', alpha=0.5)
    axes[-1].set_xlabel('Time after constant-influent step (h)')
    fig.suptitle('Steady-State Recovery under Constant Influent\n'
                  f'(Constant: COD={mean_input_real[0]:.0f}, NH4={mean_input_real[1]:.0f}, '
                  f'PO4={mean_input_real[2]:.0f}, DO={mean_input_real[3]:.1f}; thin lines = individual seeds, band = $\\pm$1 SD)')
    plt.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42  # embed fonts (Water Research §10.3)
    fig_out = os.path.join(PAPER_DIR, "fig_steady_state.pdf")
    fig.savefig(fig_out, bbox_inches='tight')                       # vector (primary)
    fig.savefig(os.path.join(PAPER_DIR, "fig_steady_state.png"),
                dpi=600, bbox_inches='tight')                       # raster backup
    plt.close(fig)
    print(f"-> {fig_out} (+ .png 600 dpi)")


if __name__ == "__main__":
    run()
