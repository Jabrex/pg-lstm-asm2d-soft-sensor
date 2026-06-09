import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from asm2d_pg_lstm import ASM2d_PG_LSTM, load_and_prepare_data

OUTPUT_DIR   = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(OUTPUT_DIR, "asm2d_pg_lstm_model.pth")
PARAMS_PATH  = os.path.join(OUTPUT_DIR, "optuna_best_params.json")
CSV_PATH     = os.path.join(OUTPUT_DIR, "synthetic_asm2d_physics.csv")
FEATURE_NAMES = ['COD Influent', 'NH4 Influent', 'PO4 Influent', 'DO']
TARGET_NAMES  = ['KOI_cikis', 'NH4_cikis', 'PO4_cikis']
TARGET_LABELS = ['COD Effluent', 'NH4 Effluent', 'PO4 Effluent']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11, 'figure.dpi': 150})


class SingleOutputWrapper(torch.nn.Module):
    def __init__(self, model, output_idx):
        super().__init__()
        self.model = model
        self.output_idx = output_idx

    def eval(self):
        return self.train()

    def forward(self, x):
        return self.model(x)[:, -1, self.output_idx].unsqueeze(1)


def load_model_and_data():
    with open(PARAMS_PATH) as f: HP = json.load(f)
    model = ASM2d_PG_LSTM(input_dim=4, hidden_dim=HP['hidden_dim'],
        num_layers=HP['num_layers'], dropout=HP['dropout'], fc_dim=HP['fc_dim']).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
    else:
        print(f"[UYARI] {MODEL_PATH} bulunamadi. Once asm2d_pg_lstm.py calistirin.")
    model.eval()
    train_loader, test_loader, scaler, _, _ = load_and_prepare_data(
        CSV_PATH, seq_len=24, batch_size=HP['batch_size'], verbose=False)
    return model, train_loader, test_loader, scaler


def collect_sequences(loader, n, device):
    seqs = []
    for X, _, _ in loader:
        seqs.append(X)
        if sum(s.shape[0] for s in seqs) >= n: break
    return torch.cat(seqs, dim=0)[:n].to(device)


def run_shap_analysis(model, train_loader, test_loader):
    try: import shap
    except ImportError:
        print("[HATA] shap yuklu degil: pip install shap")
        return
    background = collect_sequences(train_loader, 100, DEVICE)
    test_seqs  = collect_sequences(test_loader, 200, DEVICE)
    for out_idx, (tname, tlabel) in enumerate(zip(TARGET_NAMES, TARGET_LABELS)):
        print(f"  SHAP: {tname} ...")
        wrapper = SingleOutputWrapper(model, out_idx).to(DEVICE)
        wrapper.train()
        try:
            exp = shap.GradientExplainer(wrapper, background)
            sv  = exp.shap_values(test_seqs)
        except Exception as e:
            try:
                exp = shap.DeepExplainer(wrapper, background)
                sv  = exp.shap_values(test_seqs)
            except Exception as e2:
                print(f"  SHAP basarisiz ({tname}): {e2}")
                continue
        if isinstance(sv, list): sv = sv[0]
        sv = np.array(sv)
        if sv.ndim == 4: sv = sv[..., 0]
        shap_last = sv[:, -1, :]
        mean_abs = np.abs(shap_last).mean(axis=0)
        colors = ['#E63946' if float(v) >= 0 else '#457B9D' for v in shap_last.mean(axis=0)]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(FEATURE_NAMES, mean_abs, color=colors)
        for bar, val in zip(bars, mean_abs):
            ax.text(bar.get_width() + max(mean_abs)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        ax.set_xlim(0, max(mean_abs) * 1.25)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'SHAP Feature Importance — {tlabel}')
        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"shap_summary_{tname.split('_')[0].lower()}.png")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Kaydedildi: {fname}")


def run_pdp(model, test_loader, scaler):
    print("  PDP: DO -> NH4_cikis ...")
    test_seqs = collect_sequences(test_loader, 500, DEVICE)
    do_real = np.linspace(0.1, 6.0, 50)
    do_min, do_max = scaler.data_min_[3], scaler.data_max_[3]
    do_norm_vals = (do_real - do_min) / (do_max - do_min + 1e-9)
    nh4_min, nh4_max = scaler.data_min_[5], scaler.data_max_[5]
    means, stds = [], []
    model.eval()
    with torch.no_grad():
        for do_n in do_norm_vals:
            s = test_seqs.clone()
            s[:, :, 3] = float(do_n)
            nh4_n = model(s)[:, -1, 1].cpu().numpy()
            nh4_r = nh4_n * (nh4_max - nh4_min) + nh4_min
            means.append(nh4_r.mean()); stds.append(nh4_r.std())
    means, stds = np.array(means), np.array(stds)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(do_real, means, color='#E63946', linewidth=2.5, label='Mean NH4 Effluent')
    ax.fill_between(do_real, means - stds, means + stds, alpha=0.25, color='#E63946', label='±1 SD')
    ax.axvline(0.5, color='gray',      ls='--', lw=1.0, label='DO=0.5 (Nitrification Threshold)')
    ax.axvline(2.0, color='steelblue', ls='--', lw=1.0, label='DO=2.0 (Typical Setpoint)')
    ax.axhline(10,  color='orange',    ls=':',  lw=1.2, label='Operational NH4 threshold (10 mg/L)')
    ax.set_xlabel('Dissolved Oxygen — DO (mg/L)')
    ax.set_ylabel('Predicted NH4 Effluent (mg-N/L)')
    ax.set_title('Partial Dependence Plot: DO → NH4 Effluent')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "pdp_do_nh4.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Kaydedildi: {fname}")


def run_ale_1d(model, test_loader, scaler, feature_idx=3, output_idx=1,
                feature_name='DO', output_name='NH4', n_bins=20, n_samples=500):
    print(f"  ALE 1D: {feature_name} -> {output_name}_cikis ...")
    test_seqs = collect_sequences(test_loader, n_samples, DEVICE).cpu().numpy()
    n = test_seqs.shape[0]

    feat_vals = test_seqs[:, -1, feature_idx]
    quantile_edges = np.quantile(feat_vals, np.linspace(0, 1, n_bins + 1))
    quantile_edges = np.unique(quantile_edges)
    n_bins_eff = len(quantile_edges) - 1
    if n_bins_eff < 2:
        print(f"  [SKIP] Yetersiz unique bin ({n_bins_eff})")
        return

    fmin, fmax = scaler.data_min_[feature_idx], scaler.data_max_[feature_idx]
    bin_centers_real = (quantile_edges[:-1] + quantile_edges[1:]) / 2 * (fmax - fmin) + fmin

    out_scaler_idx = 4 + output_idx
    omin, omax = scaler.data_min_[out_scaler_idx], scaler.data_max_[out_scaler_idx]

    local_effects = np.zeros(n_bins_eff)
    bin_counts = np.zeros(n_bins_eff, dtype=int)

    model.eval()
    test_t = torch.tensor(test_seqs, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for b in range(n_bins_eff):
            lo, hi = quantile_edges[b], quantile_edges[b + 1]
            mask = (feat_vals >= lo) & (feat_vals < hi)
            if mask.sum() < 2:
                continue
            seqs_in_bin = test_t[mask].clone()
            seqs_lo = seqs_in_bin.clone(); seqs_lo[:, -1, feature_idx] = lo
            seqs_hi = seqs_in_bin.clone(); seqs_hi[:, -1, feature_idx] = hi
            out_lo = model(seqs_lo)[:, -1, output_idx].cpu().numpy()
            out_hi = model(seqs_hi)[:, -1, output_idx].cpu().numpy()
            local_effects[b] = float(np.mean(out_hi - out_lo))
            bin_counts[b] = int(mask.sum())

    ale_norm = np.cumsum(local_effects)
    ale_norm = ale_norm - np.mean(ale_norm)
    ale_real = ale_norm * (omax - omin)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers_real, ale_real, color='#1D3557', linewidth=2.5, marker='o', markersize=4)
    ax.fill_between(bin_centers_real, ale_real - np.std(ale_real),
                     ale_real + np.std(ale_real), alpha=0.18, color='#1D3557')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel(f'{feature_name} (mg/L)')
    ax.set_ylabel(f'ALE on {output_name} Effluent (mg/L, centered)')
    ax.set_title(f'Accumulated Local Effects: {feature_name} → {output_name} Effluent')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(bin_centers_real, bin_counts, alpha=0.15, color='gray',
             width=(bin_centers_real[1]-bin_centers_real[0]) if len(bin_centers_real) > 1 else 0.5)
    ax2.set_ylabel('Bin Sample Count', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"ale_{feature_name.lower()}_{output_name.lower()}.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Kaydedildi: {fname}")
    return bin_centers_real, ale_real


def run_ale_2d(model, test_loader, scaler, feat_idx_a=3, feat_idx_b=1,
                output_idx=1, feat_name_a='DO', feat_name_b='NH4_in',
                output_name='NH4', n_bins=10, n_samples=500):
    print(f"  ALE 2D: {feat_name_a} × {feat_name_b} -> {output_name} ...")
    test_seqs = collect_sequences(test_loader, n_samples, DEVICE).cpu().numpy()
    a_vals = test_seqs[:, -1, feat_idx_a]
    b_vals = test_seqs[:, -1, feat_idx_b]
    a_edges = np.linspace(a_vals.min(), a_vals.max(), n_bins + 1)
    b_edges = np.linspace(b_vals.min(), b_vals.max(), n_bins + 1)

    a_min, a_max = scaler.data_min_[feat_idx_a], scaler.data_max_[feat_idx_a]
    b_min, b_max = scaler.data_min_[feat_idx_b], scaler.data_max_[feat_idx_b]
    out_scaler_idx = 4 + output_idx
    omin, omax = scaler.data_min_[out_scaler_idx], scaler.data_max_[out_scaler_idx]

    grid = np.zeros((n_bins, n_bins))
    test_t = torch.tensor(test_seqs, dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for i in range(n_bins):
            for j in range(n_bins):
                seqs = test_t.clone()
                a_mid = (a_edges[i] + a_edges[i+1]) / 2
                b_mid = (b_edges[j] + b_edges[j+1]) / 2
                seqs[:, -1, feat_idx_a] = float(a_mid)
                seqs[:, -1, feat_idx_b] = float(b_mid)
                out = model(seqs)[:, -1, output_idx].cpu().numpy()
                grid[i, j] = float(np.mean(out))

    grid_real = grid * (omax - omin) + omin
    a_centers = (a_edges[:-1] + a_edges[1:]) / 2 * (a_max - a_min) + a_min
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2 * (b_max - b_min) + b_min

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid_real.T, origin='lower', aspect='auto', cmap='RdYlBu_r',
                    extent=[a_centers[0], a_centers[-1], b_centers[0], b_centers[-1]])
    ax.contour(a_centers, b_centers, grid_real.T, levels=[10.0],
                colors='black', linewidths=2, linestyles='--')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Predicted {output_name} Effluent (mg/L)')
    ax.set_xlabel(f'{feat_name_a} (mg/L)')
    ax.set_ylabel(f'{feat_name_b} (mg/L)')
    ax.set_title(f'2D Operational Map: {feat_name_a} × {feat_name_b} → {output_name}\n(black dashed = operational NH4 threshold 10 mg/L)')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"ale_2d_{feat_name_a.lower()}_{feat_name_b.lower()}_{output_name.lower()}.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Kaydedildi: {fname}")


def run_temporal_shap(model, train_loader, test_loader, output_idx=1, output_name='NH4'):
    try: import shap
    except ImportError:
        print("[HATA] shap yuklu degil")
        return
    print(f"  Temporal SHAP: {output_name} (per-lag importance) ...")
    background = collect_sequences(train_loader, 100, DEVICE)
    test_seqs  = collect_sequences(test_loader, 200, DEVICE)
    wrapper = SingleOutputWrapper(model, output_idx).to(DEVICE)
    wrapper.train()
    try:
        exp = shap.GradientExplainer(wrapper, background)
        sv = exp.shap_values(test_seqs)
    except Exception as e:
        print(f"  Temporal SHAP basarisiz: {e}")
        return
    if isinstance(sv, list): sv = sv[0]
    sv = np.array(sv)
    if sv.ndim == 4: sv = sv[..., 0]
    abs_sv = np.abs(sv).mean(axis=0)
    seq_len = abs_sv.shape[0]
    lags = np.arange(-(seq_len-1), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for f_idx, fname in enumerate(FEATURE_NAMES):
        ax.plot(lags, abs_sv[:, f_idx], marker='o', markersize=3, label=fname, linewidth=1.5)
    for k in [-24, -12, -6]:
        if k >= -(seq_len-1):
            ax.axvline(k, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Lag (hours, t=current)')
    ax.set_ylabel(f'Mean |SHAP value| for {output_name} prediction')
    ax.set_title(f'Temporal Feature Importance — {output_name} Effluent')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"temporal_shap_{output_name.lower()}.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Kaydedildi: {fname}")


def run_shap():
    print("Faz 5 — SHAP & PDP & ALE Analizi")
    model, train_loader, test_loader, scaler = load_model_and_data()
    print("\n[1/5] SHAP Feature Importance...")
    run_shap_analysis(model, train_loader, test_loader)
    print("\n[2/5] Partial Dependence Plot (legacy)...")
    run_pdp(model, test_loader, scaler)
    print("\n[3/5] ALE 1D: DO → NH4 (PDP yerine, korelasyon-robust)...")
    run_ale_1d(model, test_loader, scaler,
                feature_idx=3, output_idx=1,
                feature_name='DO', output_name='NH4')
    print("\n[4/5] ALE 2D: DO × NH4_in → NH4 effluent (operasyonel harita)...")
    run_ale_2d(model, test_loader, scaler,
                feat_idx_a=3, feat_idx_b=1, output_idx=1,
                feat_name_a='DO', feat_name_b='NH4_in', output_name='NH4')
    print("\n[5/5] Temporal SHAP: lag importance...")
    run_temporal_shap(model, train_loader, test_loader, output_idx=1, output_name='NH4')
    print("\nFaz 5 tamamlandi.")

if __name__ == "__main__":
    run_shap()
