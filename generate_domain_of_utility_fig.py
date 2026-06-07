"""
Crown Jewel Figure — Domain of Utility for PG-LSTM
=====================================================

Path B'nin görsel özü: PG-LSTM avantajı = mekanistik tutarlılık (ODE residual ratio).

Tek figure, üç panel:
  A) ODE Residual vs Training Fraction (PG-LSTM vs Vanilla)
  B) ODE Residual vs Test-time Noise σ
  C) PG-LSTM Advantage Factor = ODE_Vanilla / ODE_PG-LSTM (log-scale, all conditions)

Not: CSV'deki Model sütunu "PG_LSTM" değeri legacy isimdir; aynı modeli PG-LSTM
olarak yeniden adlandırdık. Veri filtreleri CSV ile eşleşmek için olduğu gibi
kalır; yalnızca grafik etiketleri PG-LSTM olarak güncellenir.

Çıktı:
  paper_artifacts/fig_domain_of_utility.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT      = os.path.dirname(os.path.abspath(__file__))
EXP_DIR   = os.path.join(ROOT, "experiments")
PAPER_DIR = os.path.join(ROOT, "paper_artifacts")

LOW_DATA_CSV = os.path.join(EXP_DIR, "low_data_results.csv")
NOISE_CSV    = os.path.join(EXP_DIR, "noise_robustness_results.csv")

PG_COLOR      = "#E63946"
VANILLA_COLOR = "#457B9D"


def safe_log_mean(values):
    """Outlier-robust geometric mean (NaN ve inf koru)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) == 0: return float('nan')
    return float(np.exp(np.log(v).mean()))


def main():
    # ── 1) Veriyi yükle ──────────────────────────────────────────────────────
    df_low   = pd.read_csv(LOW_DATA_CSV)
    df_noise = pd.read_csv(NOISE_CSV)

    # Low-data summary (geometric mean — büyük varyans/outlier güvenli)
    summary_low = (df_low.groupby(['Model', 'TrainFrac'])
                          ['ODE_res_mean']
                          .agg(geo_mean=safe_log_mean,
                                std=lambda x: float(np.std(x[np.isfinite(x)])))
                          .reset_index())

    # Noise summary
    summary_noise = (df_noise.groupby(['Model', 'NoiseSigma'])
                              ['ODE_res_mean']
                              .agg(geo_mean=safe_log_mean,
                                    std=lambda x: float(np.std(x[np.isfinite(x)])))
                              .reset_index())

    # ── 2) Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.4], wspace=0.32)
    ax_low   = fig.add_subplot(gs[0, 0])
    ax_noise = fig.add_subplot(gs[0, 1])
    ax_ratio = fig.add_subplot(gs[0, 2])

    # ── Panel A: Low-data ────────────────────────────────────────────────────
    fracs = sorted(df_low.TrainFrac.unique())
    x = np.arange(len(fracs))
    width = 0.36
    pg_vals = [summary_low[(summary_low.Model == "PG_LSTM") &
                                (summary_low.TrainFrac == f)].geo_mean.values[0]
                  for f in fracs]
    van_vals  = [summary_low[(summary_low.Model == "Vanilla_LSTM") &
                                (summary_low.TrainFrac == f)].geo_mean.values[0]
                  for f in fracs]
    ax_low.bar(x - width/2, pg_vals, width, color=PG_COLOR,    label='PG-LSTM',    edgecolor='black', linewidth=0.5)
    ax_low.bar(x + width/2, van_vals,  width, color=VANILLA_COLOR, label='Vanilla LSTM', edgecolor='black', linewidth=0.5)
    ax_low.set_xticks(x); ax_low.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax_low.set_yscale('log')
    ax_low.set_xlabel('Training Data Fraction')
    ax_low.set_ylabel('ASM2d ODE Residual (geo. mean, log)')
    ax_low.set_title('A — Low-Data Regime')
    ax_low.legend(loc='upper left', fontsize=9)
    ax_low.grid(True, alpha=0.3, axis='y', which='both')

    # ── Panel B: Noise ───────────────────────────────────────────────────────
    sigmas = sorted(df_noise.NoiseSigma.unique())
    x = np.arange(len(sigmas))
    pg_vals = [summary_noise[(summary_noise.Model == "PG_LSTM") &
                                  (summary_noise.NoiseSigma == s)].geo_mean.values[0]
                  for s in sigmas]
    van_vals  = [summary_noise[(summary_noise.Model == "Vanilla_LSTM") &
                                  (summary_noise.NoiseSigma == s)].geo_mean.values[0]
                  for s in sigmas]
    ax_noise.bar(x - width/2, pg_vals, width, color=PG_COLOR,    label='PG-LSTM',    edgecolor='black', linewidth=0.5)
    ax_noise.bar(x + width/2, van_vals,  width, color=VANILLA_COLOR, label='Vanilla LSTM', edgecolor='black', linewidth=0.5)
    ax_noise.set_xticks(x); ax_noise.set_xticklabels([f"σ={s:.2f}" for s in sigmas])
    ax_noise.set_yscale('log')
    ax_noise.set_xlabel('Test-time Input Noise')
    ax_noise.set_ylabel('ASM2d ODE Residual (geo. mean, log)')
    ax_noise.set_title('B — Noise Robustness')
    ax_noise.legend(loc='upper left', fontsize=9)
    ax_noise.grid(True, alpha=0.3, axis='y', which='both')

    # ── Panel C: PG-LSTM Advantage Factor (Vanilla/PG-LSTM ratio) ──────────────
    # All conditions concatenated horizontally
    cond_labels = []
    advantages  = []
    colors      = []
    for f in fracs:
        p = summary_low[(summary_low.Model == "PG_LSTM")    & (summary_low.TrainFrac == f)].geo_mean.values[0]
        v = summary_low[(summary_low.Model == "Vanilla_LSTM") & (summary_low.TrainFrac == f)].geo_mean.values[0]
        cond_labels.append(f"Data\n{int(f*100)}%")
        advantages.append(v / p if p > 0 else np.nan)
        colors.append('#2A9D8F')
    for s in sigmas:
        p = summary_noise[(summary_noise.Model == "PG_LSTM")    & (summary_noise.NoiseSigma == s)].geo_mean.values[0]
        v = summary_noise[(summary_noise.Model == "Vanilla_LSTM") & (summary_noise.NoiseSigma == s)].geo_mean.values[0]
        cond_labels.append(f"Noise\nσ={s:.2f}")
        advantages.append(v / p if p > 0 else np.nan)
        colors.append('#F4A261')

    x = np.arange(len(cond_labels))
    bars = ax_ratio.bar(x, advantages, color=colors, edgecolor='black', linewidth=0.6)
    ax_ratio.axhline(1.0, color='black', ls='--', lw=1.0, label='PG-LSTM = Vanilla')
    ax_ratio.set_yscale('log')
    ax_ratio.set_xticks(x); ax_ratio.set_xticklabels(cond_labels, fontsize=8)
    ax_ratio.set_ylabel('PG-LSTM Advantage Factor\n(ODE Residual: Vanilla / PG-LSTM)')
    ax_ratio.set_title('C — PG-LSTM Mechanistic Consistency Advantage Across Regimes')
    ax_ratio.grid(True, alpha=0.3, axis='y', which='both')
    # Etiketle her bar'a değer yaz
    for bar, val in zip(bars, advantages):
        if np.isfinite(val):
            ax_ratio.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() * 1.08,
                           f'{val:.1f}×',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    legend_elems = [
        Patch(facecolor='#2A9D8F', label='Low-data regime'),
        Patch(facecolor='#F4A261', label='Noise regime'),
    ]
    ax_ratio.legend(handles=legend_elems + [plt.Line2D([0],[0], color='black', ls='--', label='Parity')],
                     loc='upper right', fontsize=9)

    fig.suptitle("Domain of Utility for PG-LSTM in ASM2d Wastewater Modeling\n"
                  "Equivalent predictive accuracy (TOST $p<0.05$, see Tab.~\\ref{tab:tost}) "
                  "with consistently lower ASM2d ODE residual across all tested regimes",
                  fontsize=12, y=1.02)
    plt.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42  # embed fonts (Water Research §10.3)
    fig_out = os.path.join(PAPER_DIR, "fig_domain_of_utility.pdf")
    fig.savefig(fig_out, bbox_inches='tight')                       # vector (primary)
    fig.savefig(os.path.join(PAPER_DIR, "fig_domain_of_utility.png"),
                dpi=600, bbox_inches='tight')                       # raster backup
    plt.close(fig)
    print(f"-> {fig_out} (+ .png 600 dpi)")

    # ── Özet tablo (manuscript için) ─────────────────────────────────────────
    print("\n=== PG-LSTM Advantage Factor Summary ===")
    for label, adv in zip(cond_labels, advantages):
        print(f"  {label.replace(chr(10), ' '):<14}  {adv:.2f}×")


if __name__ == "__main__":
    main()
