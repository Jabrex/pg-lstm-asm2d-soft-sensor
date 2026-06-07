import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import pandas as pd
from scipy import stats

import cross_validation as cv
from cross_validation import (
    make_sequences, fold_loaders, make_model, train_fold,
    eval_fold, persistence_fold, bootstrap_ci, HP, N_FOLDS, DEVICE,
)
from equivalence_tests import diebold_mariano_test

OUTPUT_DIR = os.path.dirname(__file__)
OOD_CSV    = os.path.join(OUTPUT_DIR, "synthetic_asm2d_physics_ood.csv")


def load_ood_data():
    df = pd.read_csv(OOD_CSV)
    feature_cols = ['KOI_giris', 'NH4_giris', 'PO4_giris', 'DO']
    target_cols  = ['KOI_cikis', 'NH4_cikis', 'PO4_cikis']
    return df[feature_cols + target_cols].values.astype(np.float32)


def run_cv_ood():
    print(f"OOD CV baslatildi | Cihaz: {DEVICE} | CSV: {os.path.basename(OOD_CSV)}")
    data_raw = load_ood_data()
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
    csv_out = os.path.join(OUTPUT_DIR, "cv_results_ood.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nOOD CV sonuclari kaydedildi: {csv_out}")

    lines = ["=" * 70, "OOD (BAGIMSIZ SENARYO) ISTATISTIKSEL DOGRULAMA", "=" * 70, ""]
    metrics = ['KOI_R2', 'NH4_R2', 'PO4_R2', 'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE',
               'KOI_neg_pct', 'NH4_neg_pct', 'PO4_neg_pct',
               'ODE_res_SS', 'ODE_res_SNH', 'ODE_res_SPO4', 'ODE_res_mean']
    pg_data = df[df.Model == "PG_LSTM"]
    rivals = ["Vanilla_LSTM", "GRU", "Persistence"]

    lines.append("── Wilcoxon Signed-Rank Test (PG-LSTM vs Baselines) ──")
    lines.append(f"{'Metric':<14} {'vs Vanilla':>16} {'vs GRU':>16} {'vs Persistence':>18}")
    lines.append("-" * 66)
    for m in metrics:
        if m not in df.columns:
            continue
        row_str = f"{m:<14}"
        pg_vals = pg_data[m].values
        for rival in rivals:
            r_vals = df[df.Model == rival][m].values
            if len(pg_vals) > 1 and len(r_vals) > 1 and np.any(pg_vals - r_vals):
                try:
                    _, p = stats.wilcoxon(pg_vals, r_vals, zero_method='zsplit',
                                          alternative='two-sided')
                    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
                    row_str += f"  p={p:.4f}{sig:>2}"
                except ValueError:
                    row_str += f"  N/A           "
            else:
                row_str += f"  N/A           "
        lines.append(row_str)

    lines += ["", "── Bootstrap 95% CI (PG-LSTM, n=1000) ──",
              f"{'Metric':<14} {'Mean':>10} {'CI_low':>10} {'CI_high':>10}", "-" * 46]
    for m in metrics:
        if m not in df.columns:
            continue
        vals = pg_data[m].values
        mean, lo, hi = bootstrap_ci(vals)
        lines.append(f"{m:<14} {mean:>10.4f} {lo:>10.4f} {hi:>10.4f}")

    lines += ["", "── Model Ortalama Performans (OOD, 10-fold) ──"]
    avail = [m for m in metrics if m in df.columns]
    summary = df.groupby('Model')[avail].mean().round(4)
    lines.append(summary.to_string())

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "statistical_tests_ood.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nRapor kaydedildi: statistical_tests_ood.txt")

    try:
        dm_lines = ["", "=" * 70, "OOD DIEBOLD-MARIANO TEST (per-timestep)",
                    "Two-sided H0: equal forecast accuracy. h=1.", "=" * 70]
        for outname in ['KOI', 'NH4', 'PO4']:
            ek = f'err_{outname}'
            pg_err = np.concatenate([d[ek] for d in errors_db['PG_LSTM']])
            for rival in ['Vanilla_LSTM', 'GRU', 'Persistence']:
                rival_err = np.concatenate([d[ek] for d in errors_db[rival]])
                n = min(len(pg_err), len(rival_err))
                res = diebold_mariano_test(pg_err[:n], rival_err[:n], h=1, loss='se')
                dm_lines.append(
                    f"  {outname:4s} PG vs {rival:14s}: "
                    f"DM={res['dm_stat']:+.3f}  p={res['p_value']:.6f}  n={res['n']}"
                )
        dm_report = "\n".join(dm_lines)
        print(dm_report)
        with open(os.path.join(OUTPUT_DIR, "dm_test_results_ood.txt"), "w", encoding="utf-8") as f:
            f.write(dm_report)
    except Exception as e:
        print(f"  [WARN] OOD DM test basarisiz: {e}")


if __name__ == "__main__":
    run_cv_ood()
