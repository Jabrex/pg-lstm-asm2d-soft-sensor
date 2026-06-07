import numpy as np
import pandas as pd
from scipy import stats


def tost_test(values_a, values_b, margin, paired=True):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    if paired:
        diff = values_a - values_b
        n = len(diff)
        mean_d = float(np.mean(diff))
        sd_d   = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        se_d   = sd_d / np.sqrt(n) if n > 1 else 0.0
        df     = n - 1

        if se_d > 0:
            t_lower = (mean_d - (-margin)) / se_d
            p_lower = 1 - stats.t.cdf(t_lower, df)
            t_upper = (mean_d - margin) / se_d
            p_upper = stats.t.cdf(t_upper, df)
        else:
            p_lower = 0.0 if mean_d > -margin else 1.0
            p_upper = 0.0 if mean_d <  margin else 1.0

        if se_d > 0 and n > 1:
            t_crit = stats.t.ppf(0.975, df)
            ci_low = mean_d - t_crit * se_d
            ci_hi  = mean_d + t_crit * se_d
        else:
            ci_low = ci_hi = mean_d

        p_max = max(p_lower, p_upper)
        equivalent = p_max < 0.05
        return {
            'p_lower': float(p_lower),
            'p_upper': float(p_upper),
            'p_max':   float(p_max),
            'equivalent': bool(equivalent),
            'mean_diff': mean_d,
            'ci95':     (float(ci_low), float(ci_hi)),
            'margin':   margin,
            'n':        n,
        }
    else:
        from scipy.stats import ttest_ind
        t_l, p_l_two = ttest_ind(values_a + margin, values_b, equal_var=False)
        p_lower = p_l_two / 2 if t_l > 0 else 1 - p_l_two / 2
        t_u, p_u_two = ttest_ind(values_a - margin, values_b, equal_var=False)
        p_upper = p_u_two / 2 if t_u < 0 else 1 - p_u_two / 2
        p_max = max(p_lower, p_upper)
        return {
            'p_lower': float(p_lower), 'p_upper': float(p_upper),
            'p_max': float(p_max), 'equivalent': bool(p_max < 0.05),
            'mean_diff': float(np.mean(values_a) - np.mean(values_b)),
            'margin': margin, 'n_a': len(values_a), 'n_b': len(values_b),
        }


def diebold_mariano_test(errors_a, errors_b, h=1, loss='se'):
    errors_a = np.asarray(errors_a, dtype=float).ravel()
    errors_b = np.asarray(errors_b, dtype=float).ravel()
    if len(errors_a) != len(errors_b):
        raise ValueError("errors_a ve errors_b eşit uzunlukta olmalı")

    if loss == 'se':
        d = errors_a ** 2 - errors_b ** 2
    elif loss == 'ae':
        d = np.abs(errors_a) - np.abs(errors_b)
    else:
        raise ValueError(f"Bilinmeyen loss: {loss}")

    n = len(d)
    mean_d = float(np.mean(d))

    gamma_0 = float(np.var(d, ddof=1))
    var_d = gamma_0
    for k in range(1, h):
        w_k = 1 - k / h
        cov_k = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        var_d += 2 * w_k * cov_k

    if var_d <= 0:
        var_d = gamma_0
    se_d = np.sqrt(var_d / n)

    hln_corr = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    dm_stat = (mean_d / se_d) * hln_corr if se_d > 0 else 0.0

    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))

    return {
        'dm_stat': float(dm_stat),
        'p_value': float(p_value),
        'mean_diff': mean_d,
        'n': n,
        'h': h,
        'loss': loss,
    }


def run_tost_on_cv_results(cv_csv_path, model_a="PG_LSTM", model_b="Vanilla_LSTM",
                            metrics=None, margins=None):
    if metrics is None:
        metrics = ['KOI_R2', 'NH4_R2', 'PO4_R2', 'KOI_RMSE', 'NH4_RMSE', 'PO4_RMSE']
    if margins is None:
        margins = {
            'KOI_R2': 0.005,  'NH4_R2': 0.005, 'PO4_R2': 0.005,
            'KOI_RMSE': 0.5, 'NH4_RMSE': 0.5, 'PO4_RMSE': 0.5,
        }
    df = pd.read_csv(cv_csv_path)
    a = df[df.Model == model_a].sort_values('Fold')
    b = df[df.Model == model_b].sort_values('Fold')

    results = []
    for m in metrics:
        if m not in df.columns:
            continue
        res = tost_test(a[m].values, b[m].values, margin=margins.get(m, 0.005))
        res['metric'] = m
        res['model_a'] = model_a
        res['model_b'] = model_b
        results.append(res)
    return pd.DataFrame(results)


def write_tost_latex(tost_df, output_path, model_a="PG-LSTM"):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{TOST Equivalence Test: " + model_a + r" vs.\ baselines. "
        r"Margin $\Delta_{R^2}=0.005$, $\Delta_{\text{RMSE}}=0.5$. "
        r"$p_{\max} < 0.05$ indicates statistical equivalence within margin.}",
        r"\label{tab:tost}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Metric & Comparison & Mean Diff & 95\% CI & $p_{\max}$ & Equivalent? \\",
        r"\midrule",
    ]
    for _, r in tost_df.iterrows():
        eq = r"\checkmark" if r['equivalent'] else r"$\times$"
        lines.append(
            f"{r['metric']} & vs.\\ {r['model_b']} & "
            f"{r['mean_diff']:.4f} & "
            f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] & "
            f"{r['p_max']:.4f} & {eq} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import os
    OUT = os.path.dirname(__file__)
    cv = os.path.join(OUT, "cv_results.csv")
    if os.path.exists(cv):
        for rival in ["Vanilla_LSTM", "GRU"]:
            df = run_tost_on_cv_results(cv, "PG_LSTM", rival)
            print(f"\n── TOST: PG_LSTM vs {rival} ──")
            print(df[['metric', 'mean_diff', 'p_max', 'equivalent']].to_string(index=False))
            tex_path = os.path.join(OUT, f"tost_pg_vs_{rival.lower()}.tex")
            write_tost_latex(df, tex_path, "PG-LSTM")
            print(f"  → {tex_path}")
    else:
        print(f"cv_results.csv bulunamadı: {cv}")
