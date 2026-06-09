import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from itertools import product

from asm2d_pg_lstm import (
    ASM2d_PG_LSTM, PGLossFunction,
    load_and_prepare_data
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import ttest_rel

ABLATION_CONFIGS = {
    "Full_Model":  dict(lambda_physics=0.05,  lambda_variance=0.102, lambda_boundary=0.040),
    "No_Physics":  dict(lambda_physics=0.0,   lambda_variance=0.102, lambda_boundary=0.040),
    "No_Variance": dict(lambda_physics=0.05,  lambda_variance=0.0,   lambda_boundary=0.040),
    "No_Boundary": dict(lambda_physics=0.05,  lambda_variance=0.102, lambda_boundary=0.0),
}
SEEDS   = [42, 123, 456]
EPOCHS  = 150
PATIENCE = 30
OUTPUT_DIR = os.path.dirname(__file__)

PARAMS_PATH = os.path.join(OUTPUT_DIR, "optuna_best_params.json")
with open(PARAMS_PATH) as f:
    HP = json.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(preds_real, targets_real, names=('KOI', 'NH4', 'PO4')):
    metrics = {}
    for i, name in enumerate(names):
        y_pred = preds_real[:, i]
        y_true = targets_real[:, i]
        metrics[f"{name}_R2"]   = r2_score(y_true, y_pred)
        metrics[f"{name}_RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[f"{name}_MAPE"] = mean_absolute_percentage_error(y_true, y_pred) * 100
        metrics[f"{name}_neg_pct"] = float(np.mean(y_pred < 0) * 100)
    return metrics


def inverse_transform_outputs(preds_norm, targets_norm, scaler):
    n = scaler.n_features_in_
    dp = np.zeros((len(preds_norm), n)); dp[:, 4:7] = preds_norm
    dt = np.zeros((len(targets_norm), n)); dt[:, 4:7] = targets_norm
    return scaler.inverse_transform(dp)[:, 4:7], scaler.inverse_transform(dt)[:, 4:7]


def train_one_config(config_name, lambdas, seed, train_loader, test_loader, scaler, device):
    set_seed(seed)

    model = ASM2d_PG_LSTM(
        input_dim=4,
        hidden_dim=HP['hidden_dim'],
        num_layers=HP['num_layers'],
        dropout=HP['dropout'],
        fc_dim=HP['fc_dim']
    ).to(device)

    loss_fn = PGLossFunction(
        lambda_physics=lambdas['lambda_physics'],
        lambda_boundary=lambdas['lambda_boundary'],
        lambda_variance=lambdas['lambda_variance'],
        out_range=scaler.data_range_[4:7],
        out_min=scaler.data_min_[4:7],
    )

    optimizer = optim.AdamW(model.parameters(), lr=HP['lr'], weight_decay=HP['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_loss, patience_cnt = float('inf'), 0
    best_state = None

    NOISE_STD = HP.get('noise_std', 0.05)
    for epoch in range(EPOCHS):
        model.train()
        for X, y, raw in train_loader:
            X, y, raw = X.to(device), y.to(device), raw.to(device)
            if NOISE_STD > 0:
                scale = torch.rand(1, device=device).item() * NOISE_STD
                X = torch.clamp(X + scale * torch.randn_like(X), 0.0, 1.0)
            optimizer.zero_grad()
            preds = model(X)
            loss, *_ = loss_fn(preds, y, raw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y, raw in test_loader:
                X, y, raw = X.to(device), y.to(device), raw.to(device)
                preds = model(X)
                test_loss += loss_fn(preds, y, raw)[0].item()
        avg_test = test_loss / len(test_loader)

        if avg_test < best_loss:
            best_loss = avg_test
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y, _ in test_loader:
            preds = model(X.to(device))
            all_preds.append(preds[:, -1, :].cpu().numpy())
            all_targets.append(y[:, -1, :].cpu().numpy())

    preds_real, targets_real = inverse_transform_outputs(
        np.concatenate(all_preds), np.concatenate(all_targets), scaler
    )
    return compute_metrics(preds_real, targets_real)


def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ablation çalışması başlatıldı — Cihaz: {device}")
    print(f"Konfigürasyonlar: {list(ABLATION_CONFIGS.keys())}")
    print(f"Seed'ler: {SEEDS}")
    print(f"Toplam eğitim: {len(ABLATION_CONFIGS) * len(SEEDS)}\n")

    csv_path = os.path.join(OUTPUT_DIR, "synthetic_asm2d_physics.csv")
    train_loader, test_loader, scaler, _, _ = load_and_prepare_data(
        csv_path, seq_len=24, batch_size=HP['batch_size'], verbose=False
    )

    records = []
    for config_name, lambdas in ABLATION_CONFIGS.items():
        for seed in SEEDS:
            print(f"  [{config_name}] seed={seed} ...", end=" ", flush=True)
            row = train_one_config(
                config_name, lambdas, seed,
                train_loader, test_loader, scaler, device
            )
            row.update({'Config': config_name, 'Seed': seed})
            records.append(row)
            koi_r2 = row.get('KOI_R2', float('nan'))
            print(f"KOI_R2={koi_r2:.4f}")

    df = pd.DataFrame(records)
    csv_out = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nAblation CSV kaydedildi: {csv_out}")

    summary = df.groupby('Config').agg(['mean', 'std']).round(4)
    print("\n=== Ablation Özet (mean ± std) ===")
    for col in ['KOI_R2', 'NH4_R2', 'PO4_R2', 'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE']:
        if (col, 'mean') in summary.columns:
            print(f"  {col}: ", end="")
            for cfg in ABLATION_CONFIGS.keys():
                m = summary.loc[cfg, (col, 'mean')]
                s = summary.loc[cfg, (col, 'std')]
                print(f"{cfg}={m:.4f}±{s:.4f}  ", end="")
            print()

    _write_latex_table(df, os.path.join(OUTPUT_DIR, "ablation_table.tex"))
    print("LaTeX tablosu kaydedildi: ablation_table.tex")


def _paired_pvalue(df, metric, cfg_a, cfg_b):
    a = df[df.Config == cfg_a].sort_values('Seed')[metric].to_numpy()
    b = df[df.Config == cfg_b].sort_values('Seed')[metric].to_numpy()
    if len(a) != len(b) or len(a) < 2:
        return float('nan')
    if np.allclose(a, b):
        return 1.0
    _, p = ttest_rel(a, b)
    return float(p)


def _fmt_p(p):
    if p != p:
        return '--'
    if p < 0.001:
        return r'$p{<}0.001$'
    return rf'$p{{=}}{p:.2f}$'


def _write_latex_table(df, path):
    agg = df.groupby('Config').agg(
        KOI_R2_m=('KOI_R2', 'mean'), KOI_R2_s=('KOI_R2', 'std'),
        NH4_R2_m=('NH4_R2', 'mean'), NH4_R2_s=('NH4_R2', 'std'),
        PO4_R2_m=('PO4_R2', 'mean'), PO4_R2_s=('PO4_R2', 'std'),
        KOI_RMSE_m=('KOI_RMSE', 'mean'), KOI_RMSE_s=('KOI_RMSE', 'std'),
        NH4_RMSE_m=('NH4_RMSE', 'mean'), NH4_RMSE_s=('NH4_RMSE', 'std'),
        PO4_RMSE_m=('PO4_RMSE', 'mean'), PO4_RMSE_s=('PO4_RMSE', 'std'),
        PO4_neg_m=('PO4_neg_pct', 'mean'), PO4_neg_s=('PO4_neg_pct', 'std'),
    ).round(4)

    cfg_labels = {
        "Full_Model":  r"Full PG-LSTM",
        "No_Physics":  r"w/o Physics Loss",
        "No_Variance": r"w/o Variance Loss",
        "No_Boundary": r"w/o Boundary Loss",
    }

    pval_metrics = ['KOI_R2', 'NH4_R2', 'PO4_R2',
                    'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE']
    pvals = {}
    for cfg in cfg_labels:
        if cfg == 'Full_Model':
            continue
        pvals[cfg] = {m: _paired_pvalue(df, m, 'Full_Model', cfg)
                      for m in pval_metrics}

    lines = [
        r"\centering",
        r"\caption{Ablation study: contribution of each loss component (mean $\pm$ std across $n=3$ random seeds). Parenthetical $p$ values are paired $t$-tests against Full PG-LSTM (per-seed pairing). With $n=3$, large $p$ reflects both the absence of strong effects and the low seed count.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccccccr}",
        r"\toprule",
        r"Configuration & COD $R^2$ & NH4 $R^2$ & PO4 $R^2$ & COD RMSE & NH4 RMSE & PO4 RMSE & PO4 $<$0 (\%) \\",
        r"\midrule",
    ]
    for cfg, label in cfg_labels.items():
        if cfg not in agg.index:
            continue
        r = agg.loc[cfg]
        neg_m = float(r.PO4_neg_m) if 'PO4_neg_m' in agg.columns else 0.0
        neg_s = float(r.PO4_neg_s) if 'PO4_neg_s' in agg.columns else 0.0
        bold_open  = '\\textbf{' if neg_m == 0.0 else ''
        bold_close = '}' if neg_m == 0.0 else ''

        if cfg == 'Full_Model':
            tag = '(ref)'
            cells = [
                f"{r.KOI_R2_m:.4f}$\\pm${r.KOI_R2_s:.4f}~{tag}",
                f"{r.NH4_R2_m:.4f}$\\pm${r.NH4_R2_s:.4f}~{tag}",
                f"{r.PO4_R2_m:.4f}$\\pm${r.PO4_R2_s:.4f}~{tag}",
                f"{r.KOI_RMSE_m:.4f}$\\pm${r.KOI_RMSE_s:.4f}~{tag}",
                f"{r.NH4_RMSE_m:.4f}$\\pm${r.NH4_RMSE_s:.4f}~{tag}",
                f"{r.PO4_RMSE_m:.4f}$\\pm${r.PO4_RMSE_s:.4f}~{tag}",
            ]
        else:
            p = pvals[cfg]
            cells = [
                f"{r.KOI_R2_m:.4f}$\\pm${r.KOI_R2_s:.4f}~({_fmt_p(p['KOI_R2'])})",
                f"{r.NH4_R2_m:.4f}$\\pm${r.NH4_R2_s:.4f}~({_fmt_p(p['NH4_R2'])})",
                f"{r.PO4_R2_m:.4f}$\\pm${r.PO4_R2_s:.4f}~({_fmt_p(p['PO4_R2'])})",
                f"{r.KOI_RMSE_m:.4f}$\\pm${r.KOI_RMSE_s:.4f}~({_fmt_p(p['KOI_RMSE'])})",
                f"{r.NH4_RMSE_m:.4f}$\\pm${r.NH4_RMSE_s:.4f}~({_fmt_p(p['NH4_RMSE'])})",
                f"{r.PO4_RMSE_m:.4f}$\\pm${r.PO4_RMSE_s:.4f}~({_fmt_p(p['PO4_RMSE'])})",
            ]
        lines.append(
            f"{label} & " + " & ".join(cells) +
            f" & {bold_open}{neg_m:.2f}$\\pm${neg_s:.2f}{bold_close} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))

    print("  Paired t-test p-values (Full vs ablated):")
    for cfg, m_p in pvals.items():
        print(f"    {cfg}: " + ", ".join(f"{m}={p:.4f}" for m, p in m_p.items()))
    return pvals


if __name__ == "__main__":
    run_ablation()
