
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
from baselines.gru_model import ASM2d_GRU

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(ROOT, "synthetic_asm2d_physics.csv")
HP_PATH   = os.path.join(ROOT, "optuna_best_params.json")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(ROOT, "paper_artifacts")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS    = [42, 123, 456]
EPOCHS   = 80
PATIENCE = 15
SEQ_LEN  = 48
TEST_FRAC = 0.30

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


def make_model(model_type):
    if model_type == "PG_LSTM" or model_type == "Vanilla_LSTM":
        return ASM2d_PG_LSTM(input_dim=4, hidden_dim=HP['hidden_dim'],
            num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim'])
    elif model_type == "GRU":
        return ASM2d_GRU(input_dim=4, hidden_dim=HP['hidden_dim'],
            num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim'])


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

    model = make_model(model_type).to(DEVICE)
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


def evaluate_boundary(model, test_data, scaler):
    X_te, y_te = make_seqs(test_data)
    raw_te = X_te.copy()
    val_ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te),
                                         torch.FloatTensor(raw_te)),
        batch_size=HP['batch_size'], shuffle=False)
    model.eval()
    all_p = []
    with torch.no_grad():
        for X, y, raw in val_ld:
            p = model(X.to(DEVICE))
            all_p.append(p[:, -1, :].cpu().numpy())
    pn = np.concatenate(all_p)
    n = scaler.n_features_in_
    dp = np.zeros((len(pn), n)); dp[:, 4:7] = pn
    pr = scaler.inverse_transform(dp)[:, 4:7]

    metrics = {'N_test': len(pr)}
    for i, name in enumerate(['COD', 'NH4', 'PO4']):
        col = pr[:, i]
        n_neg = int((col < 0).sum())
        n_zero_neg = int((col < -0.01).sum())
        metrics[f'{name}_neg_count']  = n_neg
        metrics[f'{name}_neg_pct']    = 100 * n_neg / len(col)
        metrics[f'{name}_strict_neg'] = n_zero_neg
        metrics[f'{name}_min_pred']   = float(col.min())
        metrics[f'{name}_max_neg_mag']= float(-col[col < 0].min()) if n_neg > 0 else 0.0
    return metrics


def run():
    print(f"Boundary Violation Analysis | Cihaz: {DEVICE}")
    full = load_data()
    n_test = int(len(full) * TEST_FRAC)
    train_pool = full[:len(full) - n_test]
    test_data  = full[len(full) - n_test:]
    sc = MinMaxScaler(); sc.fit(train_pool)
    train_norm = sc.transform(train_pool)
    test_norm  = sc.transform(test_data)
    val_split  = train_norm[-len(test_norm)//2:]

    records = []
    for model_type in ["PG_LSTM", "Vanilla_LSTM", "GRU"]:
        for seed in SEEDS:
            print(f"\n  Eğitim: [{model_type} seed={seed}]")
            model = train_model(model_type, train_norm[:-len(val_split)], val_split, sc, seed)
            m = evaluate_boundary(model, test_norm, sc)
            row = {'Model': model_type, 'Seed': seed, **m}
            records.append(row)
            print(f"    COD neg: {m['COD_neg_count']:>4} ({m['COD_neg_pct']:.2f}%) min={m['COD_min_pred']:.3f}")
            print(f"    NH4 neg: {m['NH4_neg_count']:>4} ({m['NH4_neg_pct']:.2f}%) min={m['NH4_min_pred']:.3f}")
            print(f"    PO4 neg: {m['PO4_neg_count']:>4} ({m['PO4_neg_pct']:.2f}%) min={m['PO4_min_pred']:.3f}")

    df = pd.DataFrame(records)
    csv_out = os.path.join(OUT_DIR, "boundary_violation_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n→ {csv_out}")

    summary = df.groupby('Model').agg({
        'COD_neg_pct': ['mean', 'std'],
        'NH4_neg_pct': ['mean', 'std'],
        'PO4_neg_pct': ['mean', 'std'],
        'COD_min_pred': 'min',
        'NH4_min_pred': 'min',
        'PO4_min_pred': 'min',
    }).round(3)
    print("\n=== Boundary Violation Summary ===")
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    output_names = ['COD', 'NH4', 'PO4']
    models = ['PG_LSTM', 'Vanilla_LSTM', 'GRU']
    model_display = {'PG_LSTM': 'PG-LSTM', 'Vanilla_LSTM': 'Vanilla LSTM', 'GRU': 'GRU'}
    colors = {'PG_LSTM': '#E63946', 'Vanilla_LSTM': '#457B9D', 'GRU': '#F4A261'}

    width = 0.25; x = np.arange(len(output_names))
    for j, mt in enumerate(models):
        vals = [df[df.Model == mt][f'{o}_neg_pct'].mean() for o in output_names]
        errs = [df[df.Model == mt][f'{o}_neg_pct'].std()  for o in output_names]
        axes[0].bar(x + (j-1)*width, vals, width, yerr=errs, label=model_display[mt],
                     color=colors[mt], edgecolor='black', linewidth=0.5, capsize=3)
    axes[0].set_xticks(x); axes[0].set_xticklabels(output_names)
    axes[0].set_ylabel('% of test predictions < 0')
    axes[0].set_title('A — Boundary Violation Frequency (raw output)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    for j, mt in enumerate(models):
        vals = [-df[df.Model == mt][f'{o}_min_pred'].min() for o in output_names]
        vals = [max(v, 0.001) for v in vals]
        axes[1].bar(x + (j-1)*width, vals, width, label=model_display[mt],
                     color=colors[mt], edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(output_names)
    axes[1].set_ylabel('Most negative prediction magnitude (mg/L)')
    axes[1].set_yscale('symlog', linthresh=0.01)
    axes[1].set_title('B — Worst-Case Boundary Violation (raw output)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y', which='both')

    fig.suptitle('Boundary Loss Effect: Physically Impossible Predictions Across Models',
                  fontsize=12)
    plt.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42
    fig_out = os.path.join(PAPER_DIR, "fig_boundary_violation.pdf")
    fig.savefig(fig_out, bbox_inches='tight')
    fig.savefig(os.path.join(PAPER_DIR, "fig_boundary_violation.png"),
                dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"-> {fig_out} (+ .png 600 dpi)")

    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Raw boundary-violation frequency on the held-out test set "
        r"(mean $\pm$ std over 3 seeds). Values reported on the \emph{raw} "
        r"model output before any post-hoc clamp. Applying an inference-time "
        r"$\max(0,\hat y)$ clamp brings every model trivially to $0\%$ "
        r"violations (\cref{tab:model_comparison}); the rates below show the "
        r"underlying tendency of unconstrained data-driven models to drift "
        r"into the physically infeasible region without the training-time "
        r"boundary loss.}",
        r"\label{tab:boundary}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & COD $<$ 0 (\%) & NH4 $<$ 0 (\%) & PO4 $<$ 0 (\%) & Min PO4 pred (mg/L) \\",
        r"\midrule",
    ]
    model_tex = {'PG_LSTM': r'\PGLSTM{}', 'Vanilla_LSTM': 'Vanilla LSTM', 'GRU': 'GRU'}
    for mt in models:
        sub = df[df.Model == mt]
        cod_pct = f"{sub['COD_neg_pct'].mean():.2f}\\,$\\pm$\\,{sub['COD_neg_pct'].std():.2f}"
        nh4_pct = f"{sub['NH4_neg_pct'].mean():.2f}\\,$\\pm$\\,{sub['NH4_neg_pct'].std():.2f}"
        po4_pct = f"{sub['PO4_neg_pct'].mean():.2f}\\,$\\pm$\\,{sub['PO4_neg_pct'].std():.2f}"
        po4_min = f"{sub['PO4_min_pred'].min():.3f}"
        label   = model_tex.get(mt, mt.replace('_', '-'))
        latex_lines.append(f"{label} & {cod_pct} & {nh4_pct} & {po4_pct} & {po4_min} \\\\")
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(PAPER_DIR, "table_boundary_violation.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"→ {tex_path}")


if __name__ == "__main__":
    run()
