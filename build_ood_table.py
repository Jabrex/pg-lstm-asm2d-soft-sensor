import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(OUT, "cv_results_ood.csv"))

MODELS = ["PG_LSTM", "Vanilla_LSTM", "GRU", "Persistence"]
DISP = {"PG_LSTM": r"\PGLSTM{}", "Vanilla_LSTM": "Vanilla LSTM",
        "GRU": "GRU", "Persistence": "Persistence"}


def ms(model, col):
    v = df[df.Model == model][col].values
    return float(np.mean(v)), float(np.std(v, ddof=1))


def wilcox(col, rival):
    a = df[df.Model == "PG_LSTM"].sort_values('Fold')[col].values
    b = df[df.Model == rival].sort_values('Fold')[col].values
    if len(a) > 1 and np.any(a - b):
        try:
            _, p = stats.wilcoxon(a, b, zero_method='zsplit', alternative='two-sided')
            return p
        except ValueError:
            return float('nan')
    return float('nan')


def sig(p):
    if np.isnan(p):
        return ""
    return r"$^{**}$" if p <= 0.01 else (r"$^{*}$" if p <= 0.05 else "")


print("=== OOD CV ozet (mean +/- sd, 10 fold) ===\n")
cols_show = ['KOI_R2', 'NH4_R2', 'PO4_R2', 'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE',
             'PO4_neg_pct', 'NH4_neg_pct', 'KOI_neg_pct',
             'ODE_res_mean', 'ODE_res_mean_clamped']
for m in MODELS:
    print(f"-- {m} --")
    for c in cols_show:
        if c in df.columns:
            mu, sd = ms(m, c)
            print(f"   {c:<22} {mu:10.4f} +/- {sd:.4f}")
    print()

print("=== Wilcoxon PG vs rivals (key safety metrics) ===")
for c in ['PO4_neg_pct', 'ODE_res_mean', 'KOI_R2', 'NH4_R2', 'PO4_R2']:
    if c in df.columns:
        print(f"  {c:<14}", end="")
        for r in ["Vanilla_LSTM", "GRU", "Persistence"]:
            print(f"  vs {r}: p={wilcox(c, r):.4f}", end="")
        print()

lines = [
    r"\begin{table*}[htbp]",
    r"\centering",
    r"\caption{Independent-realisation (out-of-distribution) ten-fold "
    r"cross-validation. Models are re-trained and re-evaluated on a "
    r"second ASM2d simulator run with kinetic parameters perturbed by "
    r"$\pm15\%$ and shifted influent statistics (\cref{sec:ood}). "
    r"Mean $\pm$ sample standard deviation across folds ($n=10$). "
    r"\PGLSTM{} is the reference; $^{*}p\le0.05$, $^{**}p\le0.01$ on a "
    r"two-sided fold-level Wilcoxon signed-rank test against \PGLSTM{}.}",
    r"\label{tab:ood_comparison}",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}l*{3}{c}*{3}{c}cc@{}}",
    r"\toprule",
    r"& \multicolumn{3}{c}{$R^{2}$} & \multicolumn{3}{c}{RMSE} & "
    r"\multicolumn{2}{c}{ODE residual} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
    r"Model & COD & NH$_4$ & PO$_4$ & COD & NH$_4$ & PO$_4$ & raw & clamped \\",
    r"\midrule",
]


def row_block(m, with_sig=False):
    cells = []
    line2 = [r"\phantom{Mod.}"]
    metric_cols = ['KOI_R2', 'NH4_R2', 'PO4_R2', 'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE',
                   'ODE_res_mean', 'ODE_res_mean_clamped']
    fmt = ['%.3f', '%.3f', '%.3f', '%.2f', '%.2f', '%.2f', '%.1f', '%.1f']
    for c, f in zip(metric_cols, fmt):
        mu, sd = ms(m, c)
        star = ""
        if with_sig and m != "PG_LSTM":
            star = sig(wilcox(c, m)) if False else ""
        cells.append((f % mu) + star)
        line2.append(r"{\scriptsize $\pm " + (f % sd).lstrip() + r"$}")
    return (DISP[m] + " & " + " & ".join(cells) + r" \\",
            " & ".join(line2) + r" \\")


def row_block_sig(m):
    metric_cols = ['KOI_R2', 'NH4_R2', 'PO4_R2', 'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE',
                   'ODE_res_mean', 'ODE_res_mean_clamped']
    fmt = ['%.3f', '%.3f', '%.3f', '%.2f', '%.2f', '%.2f', '%.1f', '%.1f']
    star_cols = {'KOI_R2', 'NH4_R2', 'PO4_R2', 'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE'}
    cells, line2 = [], [r"\phantom{Mod.}"]
    for c, f in zip(metric_cols, fmt):
        mu, sd = ms(m, c)
        star = sig(wilcox(c, m)) if (m != "PG_LSTM" and c in star_cols) else ""
        cells.append((f % mu) + star)
        line2.append(r"{\scriptsize $\pm " + (f % sd).lstrip() + r"$}")
    return (DISP[m] + " & " + " & ".join(cells) + r" \\",
            " & ".join(line2) + r" \\")


for m in ["PG_LSTM", "GRU", "Vanilla_LSTM"]:
    r1, r2 = row_block_sig(m)
    lines += [r1, r2]
lines.append(r"\midrule")
r1, r2 = row_block_sig("Persistence")
lines += [r1, r2, r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

bnd = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Out-of-distribution PO$_4$-P boundary violation frequency "
    r"across ten temporal folds on the independent simulator realisation. "
    r"Parenthetical $p$ from a two-sided Wilcoxon signed-rank test against "
    r"\PGLSTM{} on the fold-wise violation rate. $^{*}p\le0.05$, "
    r"$^{**}p\le0.01$.}",
    r"\label{tab:ood_boundary}",
    r"\footnotesize",
    r"\begin{tabular}{lcc}",
    r"\toprule",
    r"Model & Violation Rate (\%) & Min PO$_4$ (mg/L) \\",
    r"\midrule",
]
for m in ["PG_LSTM", "Vanilla_LSTM", "GRU"]:
    mu, sd = ms(m, 'PO4_neg_pct')
    minp, _ = ms(m, 'PO4_min_pred')
    minval = df[df.Model == m]['PO4_min_pred'].min()
    if m == "PG_LSTM":
        tail = "(ref)"
    else:
        p = wilcox('PO4_neg_pct', m)
        star = sig(p).replace('$', '')
        tail = f"($p{{=}}{p:.3f}{star}$)" if not np.isnan(p) else ""
    bnd.append(f"{DISP[m]} & ${mu:.2f} \\pm {sd:.2f}$ {tail} & ${minval:+.3f}$ \\\\")
bnd += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

os.makedirs(os.path.join(OUT, "paper_artifacts"), exist_ok=True)
with open(os.path.join(OUT, "paper_artifacts", "table_ood.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n\n" + "\n".join(bnd) + "\n")
print("\n[OK] paper_artifacts/table_ood.tex yazildi")
