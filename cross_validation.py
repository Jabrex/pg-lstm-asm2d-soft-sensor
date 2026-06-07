import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from asm2d_pg_lstm import ASM2d_PG_LSTM, PGLossFunction
from baselines.gru_model import ASM2d_GRU
from physics_consistency import compute_ode_residual_norm

OUTPUT_DIR  = os.path.dirname(__file__)
CSV_PATH    = os.path.join(OUTPUT_DIR, "synthetic_asm2d_physics.csv")
PARAMS_PATH = os.path.join(OUTPUT_DIR, "optuna_best_params.json")
N_FOLDS     = 10
EPOCHS      = 100
PATIENCE    = 20
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(PARAMS_PATH) as f: HP = json.load(f)


def load_raw_data():
    import pandas as _pd
    from sklearn.preprocessing import MinMaxScaler
    df = _pd.read_csv(CSV_PATH)
    feature_cols = ['KOI_giris', 'NH4_giris', 'PO4_giris', 'DO']
    target_cols  = ['KOI_cikis', 'NH4_cikis', 'PO4_cikis']
    all_cols = feature_cols + target_cols
    return df[all_cols].values.astype(np.float32), df


def make_sequences(data, seq_len=48):
    X, y, raw = [], [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :4])
        y.append(data[i:i+seq_len, 4:7])
        raw.append(data[i:i+seq_len, :4])
    return np.array(X), np.array(y), np.array(raw)


def fold_loaders(X, y, raw, fold_idx, n_folds, batch_size, gap=48):
    from torch.utils.data import TensorDataset, DataLoader
    fold_size = len(X) // n_folds
    val_start = fold_idx * fold_size
    val_end   = val_start + fold_size

    train_left_end   = max(0, val_start - gap)
    train_right_start = min(len(X), val_end + gap)
    X_tr  = np.concatenate([X[:train_left_end], X[train_right_start:]])
    y_tr  = np.concatenate([y[:train_left_end], y[train_right_start:]])
    raw_tr= np.concatenate([raw[:train_left_end], raw[train_right_start:]])
    X_val, y_val, raw_val = X[val_start:val_end], y[val_start:val_end], raw[val_start:val_end]

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    n_all = np.concatenate([X_tr, raw_tr], axis=-1).reshape(-1, X_tr.shape[-1])
    all_data = np.concatenate([X_tr, y_tr], axis=-1).reshape(-1, 7)
    sc.fit(all_data)
    def norm(Xs, ys):
        combined = np.concatenate([Xs, ys], axis=-1)
        shape = combined.shape
        flat = combined.reshape(-1, 7)
        scaled = sc.transform(flat).reshape(shape)
        return scaled[:, :, :4], scaled[:, :, 4:7]

    Xn_tr, yn_tr   = norm(X_tr, y_tr)
    Xn_val, yn_val = norm(X_val, y_val)

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(Xn_tr), torch.FloatTensor(yn_tr), torch.FloatTensor(raw_tr))
    val_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(Xn_val), torch.FloatTensor(yn_val), torch.FloatTensor(raw_val))
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_ld, val_ld, sc


def make_model(model_type):
    if model_type == "PG_LSTM":
        return ASM2d_PG_LSTM(input_dim=4, hidden_dim=HP['hidden_dim'],
            num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim'])
    elif model_type == "Vanilla_LSTM":
        m = ASM2d_PG_LSTM(input_dim=4, hidden_dim=HP['hidden_dim'],
            num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim'])
        return m
    elif model_type == "GRU":
        return ASM2d_GRU(input_dim=4, hidden_dim=HP['hidden_dim'],
            num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim'])
    raise ValueError(f"Bilinmeyen model tipi: {model_type}")


def train_fold(model, model_type, train_ld, val_ld, scaler):
    model = model.to(DEVICE)
    NOISE_STD = HP.get('noise_std', 0.05)
    if model_type == "PG_LSTM":
        loss_fn_phys = PGLossFunction(
            lambda_physics=0.05, lambda_variance=0.102, lambda_boundary=0.04,
            out_range=scaler.data_range_[4:7], out_min=scaler.data_min_[4:7],
        )
        def loss_fn(p, y, r): return loss_fn_phys(p, y, r)[0]
    else:
        mse = nn.MSELoss()
        def loss_fn(p, y, r): return mse(p, y)

    opt = optim.AdamW(model.parameters(), lr=HP['lr'], weight_decay=HP['weight_decay'])
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    best_loss, patience_cnt, best_state = float('inf'), 0, None

    for epoch in range(EPOCHS):
        model.train()
        for X, y, raw in train_ld:
            X, y, raw = X.to(DEVICE), y.to(DEVICE), raw.to(DEVICE)
            if NOISE_STD > 0:
                import torch as _t
                scale = _t.rand(1, device=DEVICE).item() * NOISE_STD
                X = _t.clamp(X + scale * _t.randn_like(X), 0.0, 1.0)
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
        vl /= len(val_ld)
        if vl < best_loss:
            best_loss = vl; patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break

    model.load_state_dict(best_state)
    return model


def eval_fold(model, val_ld, scaler, _errors_sink=None):
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
    n = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 7
    dp = np.zeros((len(pn), n)); dp[:, 4:7] = pn
    dt = np.zeros((len(tn), n)); dt[:, 4:7] = tn
    pr = scaler.inverse_transform(dp)[:, 4:7]
    tr = scaler.inverse_transform(dt)[:, 4:7]
    pr_clamped = np.maximum(0.0, pr)

    out = {}
    for i, name in enumerate(['KOI', 'NH4', 'PO4']):
        out[f'{name}_R2']   = r2_score(tr[:, i], pr[:, i])
        out[f'{name}_RMSE'] = np.sqrt(mean_squared_error(tr[:, i], pr[:, i]))
        out[f'{name}_MAPE'] = mean_absolute_percentage_error(tr[:, i], pr[:, i]) * 100
        out[f'{name}_neg_pct'] = float(np.mean(pr[:, i] < 0) * 100)
        out[f'{name}_min_pred'] = float(pr[:, i].min())
        out[f'{name}_R2_clamped']   = r2_score(tr[:, i], pr_clamped[:, i])
        out[f'{name}_RMSE_clamped'] = np.sqrt(mean_squared_error(tr[:, i], pr_clamped[:, i]))

    try:
        per_state = compute_ode_residual_norm(pr, raw_last, dt=1.0, reduce='per_state')
        out['ODE_res_SS']   = per_state['SS']
        out['ODE_res_SNH']  = per_state['SNH']
        out['ODE_res_SPO4'] = per_state['SPO4']
        out['ODE_res_mean'] = compute_ode_residual_norm(pr, raw_last, dt=1.0, reduce='mean_l2')
        per_state_c = compute_ode_residual_norm(pr_clamped, raw_last, dt=1.0, reduce='per_state')
        out['ODE_res_mean_clamped'] = compute_ode_residual_norm(pr_clamped, raw_last, dt=1.0, reduce='mean_l2')
    except Exception as e:
        print(f"  [WARN] ODE residual hesabı başarısız: {e}")
        for k in ['ODE_res_SS', 'ODE_res_SNH', 'ODE_res_SPO4', 'ODE_res_mean', 'ODE_res_mean_clamped']:
            out[k] = float('nan')

    if _errors_sink is not None:
        _errors_sink.append({
            'pr': pr.astype(np.float32),
            'tr': tr.astype(np.float32),
            'err_KOI': (pr[:, 0] - tr[:, 0]).astype(np.float32),
            'err_NH4': (pr[:, 1] - tr[:, 1]).astype(np.float32),
            'err_PO4': (pr[:, 2] - tr[:, 2]).astype(np.float32),
        })

    return out


def persistence_fold(val_ld, scaler, _errors_sink=None):
    all_t, all_raw = [], []
    for _, y, raw in val_ld:
        all_t.append(y[:, -1, :].numpy())
        all_raw.append(raw[:, -1, :].numpy())
    tn = np.concatenate(all_t)
    raw_last = np.concatenate(all_raw)
    n = 7
    dt = np.zeros((len(tn), n)); dt[:, 4:7] = tn
    tr = scaler.inverse_transform(dt)[:, 4:7]
    pr = np.roll(tr, 1, axis=0); pr[0] = tr[0]
    pr_clamped = np.maximum(0.0, pr)
    out = {}
    for i, name in enumerate(['KOI', 'NH4', 'PO4']):
        out[f'{name}_R2']   = r2_score(tr[1:, i], pr[1:, i])
        out[f'{name}_RMSE'] = np.sqrt(mean_squared_error(tr[1:, i], pr[1:, i]))
        out[f'{name}_MAPE'] = mean_absolute_percentage_error(tr[1:, i], pr[1:, i]) * 100
        out[f'{name}_neg_pct'] = float(np.mean(pr[1:, i] < 0) * 100)
        out[f'{name}_min_pred'] = float(pr[1:, i].min())
        out[f'{name}_R2_clamped']   = r2_score(tr[1:, i], pr_clamped[1:, i])
        out[f'{name}_RMSE_clamped'] = np.sqrt(mean_squared_error(tr[1:, i], pr_clamped[1:, i]))
    try:
        per_state = compute_ode_residual_norm(pr, raw_last, dt=1.0, reduce='per_state')
        out['ODE_res_SS']   = per_state['SS']
        out['ODE_res_SNH']  = per_state['SNH']
        out['ODE_res_SPO4'] = per_state['SPO4']
        out['ODE_res_mean'] = compute_ode_residual_norm(pr, raw_last, dt=1.0, reduce='mean_l2')
        out['ODE_res_mean_clamped'] = compute_ode_residual_norm(pr_clamped, raw_last, dt=1.0, reduce='mean_l2')
    except Exception:
        for k in ['ODE_res_SS', 'ODE_res_SNH', 'ODE_res_SPO4', 'ODE_res_mean', 'ODE_res_mean_clamped']:
            out[k] = float('nan')

    if _errors_sink is not None:
        _errors_sink.append({
            'pr': pr.astype(np.float32),
            'tr': tr.astype(np.float32),
            'err_KOI': (pr[1:, 0] - tr[1:, 0]).astype(np.float32),
            'err_NH4': (pr[1:, 1] - tr[1:, 1]).astype(np.float32),
            'err_PO4': (pr[1:, 2] - tr[1:, 2]).astype(np.float32),
        })
    return out


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    boots = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return np.mean(values), lo, hi


def run_cv():
    print(f"Faz 6 — {N_FOLDS}-fold CV başlatıldı | Cihaz: {DEVICE}")
    data_raw, _ = load_raw_data()
    X, y, raw = make_sequences(data_raw, seq_len=48)

    MODEL_TYPES = ["PG_LSTM", "Vanilla_LSTM", "GRU", "Persistence"]
    records = []
    errors_db = {m: [] for m in MODEL_TYPES}

    for fold in range(N_FOLDS):
        print(f"\n  === Fold {fold+1}/{N_FOLDS} ===")
        train_ld, val_ld, scaler = fold_loaders(X, y, raw, fold, N_FOLDS, HP['batch_size'])

        for mtype in MODEL_TYPES:
            print(f"    [{mtype}] ...", end=" ", flush=True)
            sink = errors_db[mtype]
            if mtype == "Persistence":
                row = persistence_fold(val_ld, scaler, _errors_sink=sink)
            else:
                torch.manual_seed(42)
                model = make_model(mtype)
                model = train_fold(model, mtype, train_ld, val_ld, scaler)
                row = eval_fold(model, val_ld, scaler, _errors_sink=sink)
            row.update({'Fold': fold + 1, 'Model': mtype})
            records.append(row)
            print(f"KOI_R2={row['KOI_R2']:.4f}")

    df = pd.DataFrame(records)
    csv_out = os.path.join(OUTPUT_DIR, "cv_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nCV sonuçları kaydedildi: {csv_out}")

    lines = ["=" * 70, "FAZ 6 — İSTATİSTİKSEL DOĞRULAMA SONUÇLARI", "=" * 70, ""]
    metrics = ['KOI_R2', 'NH4_R2', 'PO4_R2', 'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE',
               'ODE_res_SS', 'ODE_res_SNH', 'ODE_res_SPO4', 'ODE_res_mean']
    pg_data  = df[df.Model == "PG_LSTM"]
    rivals = ["Vanilla_LSTM", "GRU", "Persistence"]

    lines.append("── Wilcoxon Signed-Rank Test (PG-LSTM vs Baselines) ──")
    lines.append(f"{'Metric':<12} {'vs Vanilla':>14} {'vs GRU':>14} {'vs Persistence':>16}")
    lines.append("-" * 58)
    for m in metrics:
        row_str = f"{m:<12}"
        pg_vals = pg_data[m].values
        for rival in rivals:
            r_vals = df[df.Model == rival][m].values
            if len(pg_vals) > 1 and len(r_vals) > 1:
                _, p = stats.wilcoxon(pg_vals, r_vals, zero_method='zsplit', alternative='two-sided')
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
                row_str += f"  p={p:.4f}{sig:>2}"
            else:
                row_str += f"  N/A           "
        lines.append(row_str)

    lines += ["", "── Bootstrap 95% CI (PG-LSTM, n=1000 bootstrap) ──",
              f"{'Metric':<12} {'Mean':>10} {'CI_low':>10} {'CI_high':>10}"]
    lines.append("-" * 44)
    for m in metrics:
        vals = pg_data[m].values
        mean, lo, hi = bootstrap_ci(vals)
        lines.append(f"{m:<12} {mean:>10.4f} {lo:>10.4f} {hi:>10.4f}")

    lines += ["", "── Model Ortalama Performans (5-fold) ──"]
    summary = df.groupby('Model')[metrics].mean().round(4)
    lines.append(summary.to_string())

    report = "\n".join(lines)
    print("\n" + report)
    report_path = os.path.join(OUTPUT_DIR, "statistical_tests.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nRapor kaydedildi: {report_path}")

    from equivalence_tests import diebold_mariano_test
    try:
        dm_lines = ["", "=" * 70,
                    "DIEBOLD-MARIANO TEST (per-timestep, paired errors)",
                    "Two-sided H0: equal forecast accuracy. h=1 (one-step).", "=" * 70]
        for outname in ['KOI', 'NH4', 'PO4']:
            ek = f'err_{outname}'
            pg_err = np.concatenate([d[ek] for d in errors_db['PG_LSTM']])
            for rival in ['Vanilla_LSTM', 'GRU', 'Persistence']:
                rival_err = np.concatenate([d[ek] for d in errors_db[rival]])
                n = min(len(pg_err), len(rival_err))
                res = diebold_mariano_test(pg_err[:n], rival_err[:n], h=1, loss='se')
                dm_lines.append(
                    f"  {outname:4s} PG-LSTM vs {rival:14s}: "
                    f"DM={res['dm_stat']:+.3f}  p={res['p_value']:.6f}  n={res['n']}"
                )
        dm_report = "\n".join(dm_lines)
        print(dm_report)
        with open(os.path.join(OUTPUT_DIR, "dm_test_results.txt"), "w", encoding="utf-8") as f:
            f.write(dm_report)

        latex_dm = [
            r"\centering",
            r"\caption{Diebold-Mariano test (one-step-ahead squared-error loss, paired per-timestep, "
            r"two-sided). $H_0$: equal forecast accuracy. Computed on the concatenated test-fold "
            r"predictions across all " + str(N_FOLDS) + r" folds ($n>10^4$), which gives orders-of-magnitude "
            r"more statistical power than the fold-level Wilcoxon test.}",
            r"\label{tab:dm}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Output & Comparison & DM stat & $p$ & Direction & Equal-accuracy? \\",
            r"\midrule",
        ]
        for outname in ['KOI', 'NH4', 'PO4']:
            ek = f'err_{outname}'
            pg_err = np.concatenate([d[ek] for d in errors_db['PG_LSTM']])
            disp_out = {'KOI':'COD','NH4':r'NH$_4$','PO4':r'PO$_4$'}[outname]
            for rival, rdisp in [('Vanilla_LSTM','Vanilla LSTM'),('GRU','GRU'),('Persistence','Persistence')]:
                rival_err = np.concatenate([d[ek] for d in errors_db[rival]])
                n = min(len(pg_err), len(rival_err))
                res = diebold_mariano_test(pg_err[:n], rival_err[:n], h=1, loss='se')
                p = res['p_value']
                direction = r'PG $<$ rival' if res['dm_stat'] < 0 else r'PG $>$ rival'
                eq = r'$\checkmark$' if p > 0.05 else r'$\times$'
                p_str = f"${p:.3f}$" if p > 1e-3 else r"$<10^{-3}$"
                latex_dm.append(
                    f"{disp_out} & vs.\\ {rdisp} & ${res['dm_stat']:+.2f}$ & {p_str} & {direction} & {eq} \\\\"
                )
        latex_dm += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(OUTPUT_DIR, "paper_artifacts", "table_dm.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(latex_dm))
        print(f"\nDM tablo yazıldı: paper_artifacts/table_dm.tex")
    except Exception as e:
        print(f"  [WARN] DM test başarısız: {e}")


if __name__ == "__main__":
    run_cv()
