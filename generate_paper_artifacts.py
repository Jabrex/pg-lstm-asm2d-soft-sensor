"""
Faz 7 — Paper Artifacts Generator
Outputs: paper_artifacts/ (3 LaTeX tables + 6 publication figures)
"""

import os, sys, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'paper_artifacts')
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 11,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'figure.dpi': 150,
    'pdf.fonttype': 42,   # embed TrueType fonts in PDF (Water Research §10.3)
    'ps.fonttype': 42,    # same for EPS
})

SEQ_LEN = 24
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_ORDER  = ['PG_LSTM', 'GRU', 'Vanilla_LSTM', 'Persistence']
MODEL_LABELS = {'PG_LSTM': 'PG-LSTM', 'GRU': 'GRU',
                'Vanilla_LSTM': 'Vanilla LSTM', 'Persistence': 'Persistence'}
OUT_LABELS   = ['COD Effluent (mg/L)', 'NH4 Effluent (mg-N/L)', 'PO4 Effluent (mg-P/L)']

# ── helpers ──────────────────────────────────────────────────────────────────
def _save(fig, name):
    # Emit a vector PDF (primary, resolution-independent, fonts embedded) plus a
    # 600 dpi PNG backup. Water Research §10.3 prefers vector for line art and
    # requires >=500 dpi for line/halftone combinations.
    base = os.path.splitext(name)[0]
    fig.savefig(os.path.join(OUTPUT_DIR, base + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, base + '.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {base}.pdf (+ {base}.png 600 dpi)')


def _load_pg():
    from asm2d_pg_lstm import ASM2d_PG_LSTM
    with open(os.path.join(BASE_DIR, 'optuna_best_params.json')) as f:
        HP = json.load(f)
    m = ASM2d_PG_LSTM(4, HP['hidden_dim'], HP['num_layers'],
                         HP['dropout'], HP['fc_dim']).to(DEVICE)
    ckpt = torch.load(os.path.join(BASE_DIR, 'asm2d_pg_lstm_model.pth'),
                      map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m, HP


def _load_gru(HP):
    from baselines.gru_model import ASM2d_GRU
    m = ASM2d_GRU(4, HP['hidden_dim'], HP['num_layers'],
                   HP['dropout'], HP['fc_dim']).to(DEVICE)
    sd = torch.load(os.path.join(BASE_DIR, 'baselines', 'gru_model.pth'),
                    map_location=DEVICE, weights_only=False)
    m.load_state_dict(sd)
    m.eval()
    return m


def _load_vanilla(HP):
    from asm2d_pg_lstm import ASM2d_PG_LSTM
    m = ASM2d_PG_LSTM(4, HP['hidden_dim'], HP['num_layers'],
                         HP['dropout'], HP['fc_dim']).to(DEVICE)
    sd = torch.load(os.path.join(BASE_DIR, 'baselines', 'vanilla_lstm_model.pth'),
                    map_location=DEVICE, weights_only=False)
    m.load_state_dict(sd)
    m.eval()
    return m


def _get_preds(model, loader, scaler):
    all_p, all_t = [], []
    with torch.no_grad():
        for X, y, _ in loader:
            p = model(X.to(DEVICE))
            all_p.append(p[:, -1, :].cpu().numpy())
            all_t.append(y[:, -1, :].cpu().numpy())
    pn = np.concatenate(all_p)
    tn = np.concatenate(all_t)
    n  = scaler.n_features_in_
    dp = np.zeros((len(pn), n)); dp[:, 4:7] = pn
    dt = np.zeros((len(tn), n)); dt[:, 4:7] = tn
    pr = scaler.inverse_transform(dp)[:, 4:7]
    tr = scaler.inverse_transform(dt)[:, 4:7]
    return pr, tr


# ── Table 1: Model Comparison (CV mean ± std) ────────────────────────────────
def table1():
    print('[Table 1] Model comparison...')
    df = pd.read_csv(os.path.join(BASE_DIR, 'cv_results.csv'))
    agg = df.groupby('Model').agg(
        KOI_R2_m=('KOI_R2','mean'), KOI_R2_s=('KOI_R2','std'),
        NH4_R2_m=('NH4_R2','mean'), NH4_R2_s=('NH4_R2','std'),
        PO4_R2_m=('PO4_R2','mean'), PO4_R2_s=('PO4_R2','std'),
        KOI_RMSE_m=('KOI_RMSE','mean'), KOI_RMSE_s=('KOI_RMSE','std'),
        NH4_RMSE_m=('NH4_RMSE','mean'), NH4_RMSE_s=('NH4_RMSE','std'),
        PO4_RMSE_m=('PO4_RMSE','mean'), PO4_RMSE_s=('PO4_RMSE','std'),
    ).round(4)

    def fmt(m, s, bold=False):
        v = f'{m:.4f}$\\pm${s:.4f}'
        return f'\\textbf{{{v}}}' if bold else v

    lines = [
        r'\begin{table}[ht]', r'\centering',
        r'\caption{Performance comparison across 5-fold temporal cross-validation '
        r'(mean $\pm$ std). PG-LSTM achieves statistically equivalent accuracy '
        r'to data-driven baselines (Wilcoxon $p>0.05$, see Table~\ref{tab:wilcoxon}).}',
        r'\label{tab:model_comparison}',
        r'\begin{tabular}{lcccccc}', r'\toprule',
        r'Model & COD $R^2$ & NH4 $R^2$ & PO4 $R^2$ & COD RMSE & NH4 RMSE & PO4 RMSE \\',
        r'& & & & (mg/L) & (mg-N/L) & (mg-P/L) \\',
        r'\midrule',
    ]
    for mk in MODEL_ORDER:
        if mk not in agg.index:
            continue
        r = agg.loc[mk]
        label = MODEL_LABELS[mk]
        is_pg = (mk == 'PG_LSTM')
        if mk == 'Persistence':
            lines.append(r'\midrule')
        lines.append(
            f'{label} & {fmt(r.KOI_R2_m,r.KOI_R2_s,is_pg)} & '
            f'{fmt(r.NH4_R2_m,r.NH4_R2_s,is_pg)} & '
            f'{fmt(r.PO4_R2_m,r.PO4_R2_s,is_pg)} & '
            f'{fmt(r.KOI_RMSE_m,r.KOI_RMSE_s,is_pg)} & '
            f'{fmt(r.NH4_RMSE_m,r.NH4_RMSE_s,is_pg)} & '
            f'{fmt(r.PO4_RMSE_m,r.PO4_RMSE_s,is_pg)} \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    out = os.path.join(OUTPUT_DIR, 'table1_model_comparison.tex')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Saved: table1_model_comparison.tex')


# ── Table 2: Ablation (already generated, copy & reformat) ──────────────────
def table2():
    print('[Table 2] Ablation...')
    src = os.path.join(BASE_DIR, 'ablation_table.tex')
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUTPUT_DIR, 'table2_ablation.tex'))
        print('  Saved: table2_ablation.tex')
    else:
        print('  SKIP: ablation_table.tex not found')


# ── Table 3: Wilcoxon p-values ───────────────────────────────────────────────
def table3():
    print('[Table 3] Wilcoxon tests...')
    df = pd.read_csv(os.path.join(BASE_DIR, 'cv_results.csv'))
    metrics  = ['KOI_R2', 'NH4_R2', 'PO4_R2', 'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE']
    display  = ['COD $R^2$', 'NH4 $R^2$', 'PO4 $R^2$',
                'COD RMSE', 'NH4 RMSE', 'PO4 RMSE']
    rivals   = ['Vanilla_LSTM', 'GRU', 'Persistence']
    rlabels  = ['Vanilla LSTM', 'GRU', 'Persistence']
    pg_df  = df[df.Model == 'PG_LSTM']

    n_folds = pg_df.shape[0]
    lines = [
        r'\centering',
        r'\caption{Wilcoxon signed-rank test: PG-LSTM vs. baselines '
        rf'($n={n_folds}$ folds). $^{{*}}$: $p<0.05$; $^{{**}}$: $p<0.01$; ns: $p\geq0.05$.}}',
        r'\label{tab:wilcoxon}',
        r'\begin{tabular}{lccc}', r'\toprule',
        r'Metric & vs.\ Vanilla LSTM & vs.\ GRU & vs.\ Persistence \\',
        r'\midrule',
    ]
    for m, disp in zip(metrics, display):
        parts = []
        pv = pg_df[m].values
        for rv in rivals:
            rv_vals = df[df.Model == rv][m].values
            try:
                _, p = stats.wilcoxon(pv, rv_vals, zero_method='zsplit')
                sig = r'$^{**}$' if p < 0.01 else (r'$^{*}$' if p < 0.05 else 'ns')
                parts.append(f'$p={p:.4f}${sig}')
            except Exception:
                parts.append('N/A')
        lines.append(disp + ' & ' + ' & '.join(parts) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    out = os.path.join(OUTPUT_DIR, 'table3_wilcoxon.tex')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('  Saved: table3_wilcoxon.tex')


# ── Figure 1: A2O Reactor Schematic ──────────────────────────────────────────
def fig1():
    print('[Figure 1] Reactor schematic...')
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')

    zones = [
        (0.5,  1.5, 2.5, 3.2, 'Anaerobic Zone', '#7C3AED',
         'AN\nV = 1500 m³\nDO = 0 mg/L'),
        (4.0,  1.5, 2.5, 3.2, 'Anoxic Zone',    '#D97706',
         'AX\nV = 1500 m³\nDO ≈ 0 mg/L'),
        (7.5,  1.5, 3.0, 3.2, 'Aerobic Zone',   '#0EA5E9',
         'AE\nV = 3000 m³\nDO = 2.5 mg/L'),
        (11.5, 1.5, 1.8, 3.2, 'Settler',        '#6B7280', 'Settler'),
    ]
    for x, y, w, h, title, col, desc in zones:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.12',
            facecolor=col, alpha=0.18, edgecolor=col, linewidth=2.4))
        ax.text(x+w/2, y+h-0.30, title, ha='center', va='top',
                fontsize=14, fontweight='bold', color=col)
        ax.text(x+w/2, y+h/2-0.20, desc, ha='center', va='center',
                fontsize=12.5, color='#222')

    def arr(x1, y1, x2, y2, col='#374151', rad=0):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.8,
                                   connectionstyle=f'arc3,rad={rad}',
                                   mutation_scale=16))

    # Main flow
    arr(0.1, 3.1, 0.5, 3.1); ax.text(0.05, 3.45, 'Influent Q', fontsize=12)
    arr(3.0, 3.1, 4.0, 3.1); ax.text(3.30, 3.45, 'Q+Q_RAS',  fontsize=12)
    arr(6.5, 3.1, 7.5, 3.1); ax.text(6.75, 3.45, 'Q_total',  fontsize=12)
    arr(10.5,3.1,11.5,3.1)
    arr(13.3,3.1,13.9,3.1); ax.text(12.85,3.45,'Effluent Q', fontsize=12)

    # Q_RAS bottom (Settler → AN)
    ax.plot([12.4,12.4],[1.5,0.7], color='#DC2626', lw=2.0)
    ax.plot([12.4,1.75],[0.7,0.7], color='#DC2626', lw=2.0)
    arr(1.75,0.7,1.75,1.5, col='#DC2626')
    ax.text(7.0, 0.25, 'Return Activated Sludge  Q_RAS = Q',
            ha='center', fontsize=13, fontweight='bold', color='#DC2626')

    # Q_IR top (AE → AX)
    ax.plot([9.0,9.0],[4.7,5.4], color='#0369A1', lw=2.0)
    ax.plot([9.0,5.25],[5.4,5.4], color='#0369A1', lw=2.0)
    arr(5.25,5.4,5.25,4.7, col='#0369A1')
    ax.text(7.1, 5.80, 'Internal Recycle  Q_IR = 3Q',
            ha='center', fontsize=13, fontweight='bold', color='#0369A1')

    ax.set_title('Multi-zone A²O Reactor (ASM2d, 3 zones × 18 states = 54 ODEs)',
                 fontsize=16, fontweight='bold', pad=14)
    plt.tight_layout()
    _save(fig, 'fig1_reactor_schematic.png')


# ── Figure 2: Loss Curve ─────────────────────────────────────────────────────
def fig2():
    # Superseded by regen_loss_curve_from_wandb.py, which redraws the loss curve
    # as a vector PDF from the saved wandb history (the per-epoch history is not
    # in the checkpoint, and the old loss_curve.png was only 200 dpi). We do NOT
    # copy the 200 dpi PNG here so it cannot clobber the vector output.
    pdf = os.path.join(OUTPUT_DIR, 'fig2_loss_curve.pdf')
    print('[Figure 2] Loss curve...')
    if os.path.exists(pdf):
        print('  OK: fig2_loss_curve.pdf present (run regen_loss_curve_from_wandb.py to refresh)')
    else:
        print('  WARN: fig2_loss_curve.pdf missing - run regen_loss_curve_from_wandb.py')


# ── Figure 3: Scatter True vs Predicted ──────────────────────────────────────
def fig3():
    print('[Figure 3] Scatter plots...')
    from asm2d_pg_lstm import load_and_prepare_data

    with open(os.path.join(BASE_DIR, 'optuna_best_params.json')) as f:
        HP = json.load(f)
    csv_path = os.path.join(BASE_DIR, 'synthetic_asm2d_physics.csv')
    _, test_ld, scaler, _, _ = load_and_prepare_data(
        csv_path, seq_len=SEQ_LEN, batch_size=HP['batch_size'], verbose=False)

    pg, HP = _load_pg()
    gru       = _load_gru(HP)
    vanilla   = _load_vanilla(HP)

    pr_pg,   tr = _get_preds(pg,    test_ld, scaler)
    pr_gru,    _  = _get_preds(gru,     test_ld, scaler)
    pr_vanilla, _ = _get_preds(vanilla, test_ld, scaler)

    models = [
        ('PG-LSTM',    pr_pg,   '#2563EB'),
        ('GRU',          pr_gru,    '#16A34A'),
        ('Vanilla LSTM', pr_vanilla,'#D97706'),
    ]
    short_labels = ['COD Eff. (mg/L)', 'NH4 Eff. (mg-N/L)', 'PO4 Eff. (mg-P/L)']

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for col, (mname, preds, color) in enumerate(models):
        for row, slbl in enumerate(short_labels):
            ax = axes[row, col]
            yt = tr[:, row]; yp = preds[:, row]
            lo = min(yt.min(), yp.min()) * 0.97
            hi = max(yt.max(), yp.max()) * 1.03
            ax.scatter(yt, yp, alpha=0.25, s=4, color=color, rasterized=True)
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0)
            r2 = r2_score(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            ax.text(0.04, 0.93, f'$R^2$={r2:.4f}\nRMSE={rmse:.2f}',
                    transform=ax.transAxes, fontsize=8.5, va='top')
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel('Observed', fontsize=9)
            ax.set_ylabel('Predicted', fontsize=9)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(mname, fontsize=10, fontweight='bold', color=color)
            if col == 0:
                ax.set_ylabel(f'{slbl}\nPredicted', fontsize=8.5)

    plt.suptitle('True vs. Predicted — Test Set', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'fig3_scatter.png')


# ── Figures 4 & 5: SHAP + PDP (copy) ─────────────────────────────────────────
def fig45():
    print('[Figures 4 & 5] SHAP + PDP...')
    copies = [
        ('shap_summary_koi.png', 'fig4a_shap_cod.png'),
        ('shap_summary_nh4.png', 'fig4b_shap_nh4.png'),
        ('shap_summary_po4.png', 'fig4c_shap_po4.png'),
        ('pdp_do_nh4.png',       'fig5_pdp_do_nh4.png'),
    ]
    # Copy both the vector PDF (preferred) and the PNG backup when present, so the
    # LaTeX extensionless \includegraphics picks the PDF.
    for src_name, dst_name in copies:
        src_base = os.path.splitext(src_name)[0]
        dst_base = os.path.splitext(dst_name)[0]
        for ext in ('.pdf', '.png'):
            src = os.path.join(BASE_DIR, src_base + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(OUTPUT_DIR, dst_base + ext))
                print(f'  Saved: {dst_base + ext}')


# ── Figure 6: Time Series (14 days) ──────────────────────────────────────────
def fig6():
    print('[Figure 6] Time series...')
    from asm2d_pg_lstm import load_and_prepare_data

    pg, HP = _load_pg()
    csv_path = os.path.join(BASE_DIR, 'synthetic_asm2d_physics.csv')
    _, test_ld, scaler, _, _ = load_and_prepare_data(
        csv_path, seq_len=SEQ_LEN, batch_size=HP['batch_size'], verbose=False)

    pr, tr = _get_preds(pg, test_ld, scaler)

    show = min(336, len(tr))   # 14 days = 336 h
    t    = np.arange(show)
    colors_obs  = ['#1D4ED8', '#15803D', '#B45309']
    colors_pred = ['#93C5FD', '#86EFAC', '#FCD34D']
    short = ['COD Effluent (mg/L)', 'NH4 Effluent (mg-N/L)', 'PO4 Effluent (mg-P/L)']

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for i, (ax, lbl, co, cp) in enumerate(zip(axes, short, colors_obs, colors_pred)):
        ax.plot(t, tr[:show, i], color=co, lw=1.3, label='Observed', alpha=0.95)
        ax.plot(t, pr[:show, i], color=cp, lw=1.3, label='PG-LSTM', ls='--', alpha=0.9)
        ax.set_ylabel(lbl, fontsize=10)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.7)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel('Time [h]', fontsize=11)
    plt.suptitle('PG-LSTM: Observed vs. Predicted — First 14 Days (Test Set)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'fig6_timeseries.png')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('  Faz 7 — Generating Paper Artifacts')
    print(f'  Output: {OUTPUT_DIR}')
    print('=' * 60)

    table1()
    table2()
    table3()
    fig1()
    fig2()
    try:
        fig3()
    except Exception as e:
        print(f'  [Figure 3] ERROR: {e}')
    fig45()
    try:
        fig6()
    except Exception as e:
        print(f'  [Figure 6] ERROR: {e}')

    print()
    print('Done!')
    files = sorted(os.listdir(OUTPUT_DIR))
    print(f'paper_artifacts/ ({len(files)} files):')
    for f in files:
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
        print(f'  {f:40s} {sz:6d} KB')
