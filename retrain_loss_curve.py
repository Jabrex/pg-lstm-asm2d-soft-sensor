"""
fig2_loss_curve'u GUNCEL kodla (directional process-consistency physics
loss) yeniden uret. Eski wandb run (run-qncawbt6, 2026-05-07) loss
yeniden tanimlanmadan ONCEki sabit ~6.15 physics terimini loglamis;
figur o bayat seriyi cizyordu ve caption (directional penalty, data'nin
altinda) ile celisiyordu.

Bu script main() egitim konfigurasyonunu BIREBIR tekrarlar
(seq_len=24, epochs=200, patience=85, Optuna HP), train_model'in
dondurdugu gercek per-epoch history'yi alir ve fig2_loss_curve'u
yeniden cizer. URETIM ARTEFAKTLARINI (model.pth / scaler.joblib /
diger figurler) OVERWRITE ETMEZ — sadece paper_artifacts/fig2_loss_curve.*
yazilir.
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from asm2d_pg_lstm import (
    ASM2d_PG_LSTM, load_and_prepare_data, train_model, FIXED_SEQ_LEN,
)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'paper_artifacts')
CSV = os.path.join(BASE, 'synthetic_asm2d_physics.csv')

plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 11,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

with open(os.path.join(BASE, 'optuna_best_params.json')) as f:
    HP = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f'Device: {device} | seq_len={FIXED_SEQ_LEN} | reproduce main() training')

train_loader, test_loader, scaler, _, _ = load_and_prepare_data(
    CSV, seq_len=FIXED_SEQ_LEN, train_ratio=0.8,
    batch_size=HP['batch_size'], verbose=True,
)

model = ASM2d_PG_LSTM(
    input_dim=4, hidden_dim=HP['hidden_dim'], num_layers=HP['num_layers'],
    dropout=HP['dropout'], fc_dim=HP['fc_dim'],
)

model, history, best_val = train_model(
    model, train_loader, test_loader, scaler, device,
    epochs=200, lr=HP['lr'], weight_decay=HP['weight_decay'],
    patience=85, lambda_phys_max=HP['lambda_phys_max'],
    lambda_boundary=HP['lambda_boundary'],
    lambda_variance=HP.get('lambda_variance', 0.30),
    noise_std=HP['noise_std'], verbose=True, use_wandb=False,
)

tr_tot = np.array(history['train_total'])
te_tot = np.array(history['test_total'])
tr_dat = np.array(history['train_data'])
tr_phy = np.array(history['train_physics'])
n = len(tr_tot)
ep = range(1, n + 1)

print('\n── Gercek (directional) physics loss tani ──')
print(f'  epochs={n}')
print(f'  data  : first={tr_dat[0]:.5g} last={tr_dat[-1]:.5g} min={tr_dat.min():.5g}')
print(f'  phys  : first={tr_phy[0]:.5g} last={tr_phy[-1]:.5g} '
      f'min={tr_phy.min():.5g} max={tr_phy.max():.5g}')
print(f'  total : first={tr_tot[0]:.5g} last={tr_tot[-1]:.5g}')
ratio = tr_dat[-1] / max(tr_phy[-1], 1e-12)
print(f'  data/phys (last) = {ratio:.1f}x  -> phys, data\'nin ~{np.log10(ratio):.1f} '
      f'kat-buyukluk {"altinda" if ratio>1 else "ustunde"}')

fig = plt.figure(figsize=(8, 6))
plt.semilogy(ep, tr_tot, 'b-', alpha=0.8, label='Train (Total)', linewidth=1.5)
plt.semilogy(range(1, len(te_tot) + 1), te_tot, 'r-', alpha=0.8,
             label='Test (Total)', linewidth=1.5)
plt.semilogy(range(1, len(tr_dat) + 1), tr_dat, 'b--', alpha=0.5,
             label='Train (Data)', linewidth=1.0)
plt.semilogy(range(1, len(tr_phy) + 1), np.maximum(tr_phy, 1e-12), 'g--', alpha=0.5,
             label='Physics Loss', linewidth=1.0)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss (log scale)', fontsize=11)
plt.title('Training and Validation Loss Curve', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs(OUT, exist_ok=True)
pdf = os.path.join(OUT, 'fig2_loss_curve.pdf')
png = os.path.join(OUT, 'fig2_loss_curve.png')
fig.savefig(pdf, bbox_inches='tight')
fig.savefig(png, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {pdf}')
print(f'Saved: {png} (600 dpi)')
print('NOT: model.pth / scaler.joblib / diger figurler DOKUNULMADI.')
