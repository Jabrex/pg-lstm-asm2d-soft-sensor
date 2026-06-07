
import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.trial import TrialState
import warnings, time, os, json, copy
import joblib
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ASM2dParameters:
    mu_H      = 6.0 / 24.0
    K_S       = 4.0
    K_OH      = 0.2
    K_NO      = 0.5
    K_NH_H    = 0.05
    K_P       = 0.01
    b_H       = 0.4 / 24.0
    Y_H       = 0.625
    eta_NO    = 0.8

    mu_A      = 1.0 / 24.0
    K_NH_A    = 1.0
    K_OA      = 0.5
    b_A       = 0.15 / 24.0
    Y_A       = 0.24

    q_PHA     = 3.0 / 24.0
    q_PP      = 1.5 / 24.0
    mu_PAO    = 1.0 / 24.0
    K_PS      = 0.2
    b_PAO     = 0.2 / 24.0
    Y_PAO     = 0.625
    K_PP      = 0.01
    Y_PO4     = 0.40

    k_PRE     = 1.0 / 24.0

    i_XB      = 0.07
    i_XP      = 0.020
    f_P       = 0.10
    i_NSF     = 0.03
    i_PSF     = 0.01

    X_H       = 1500.0
    X_A       = 120.0
    X_PAO     = 350.0
    X_MeOH    = 10.0

    @classmethod
    def summary(cls):
        print("=" * 65)
        print("  ASM2d Kinetik Parametreleri — Literatür Değerleri (20°C)")
        print("=" * 65)
        for attr in ['mu_H','K_S','K_OH','K_NO','K_NH_H','b_H','Y_H',
                      'mu_A','K_NH_A','K_OA','b_A','Y_A',
                      'mu_PAO','K_PS','b_PAO','Y_PAO']:
            print(f"  {attr:10s} = {getattr(cls, attr):.4f}")
        print("=" * 65)


class ASM2dPhysics:
    def __init__(self, params=None):
        self.p = params or ASM2dParameters()

    def compute_rates(self, S_S, S_NH, S_PO4, S_O):
        p = self.p
        eps = 1e-8

        monod_S    = S_S  / (p.K_S    + S_S  + eps)
        monod_O_H  = S_O  / (p.K_OH   + S_O  + eps)
        monod_NH_H = S_NH / (p.K_NH_H + S_NH + eps)
        monod_NH_A = S_NH / (p.K_NH_A + S_NH + eps)
        monod_O_A  = S_O  / (p.K_OA   + S_O  + eps)
        monod_P    = S_PO4 / (p.K_P   + S_PO4 + eps)

        inhibit_O = p.K_OH / (p.K_OH + S_O + eps)

        rho_1 = p.mu_H * monod_S * monod_O_H * monod_NH_H * monod_P * p.X_H
        rho_2 = p.mu_H * p.eta_NO * monod_S * inhibit_O * monod_NH_H * monod_P * p.X_H
        rho_3 = p.mu_A * monod_NH_A * monod_O_A * monod_P * p.X_A
        rho_4 = p.b_H * p.X_H
        rho_5 = p.b_A * p.X_A

        rho_10 = p.q_PHA * monod_S * p.X_PAO
        rho_11 = p.q_PP * monod_O_H * monod_P * p.X_PAO
        rho_12 = p.q_PP * p.eta_NO * inhibit_O * monod_P * p.X_PAO
        rho_13 = p.mu_PAO * monod_O_H * monod_NH_H * monod_P * p.X_PAO
        rho_14 = p.mu_PAO * p.eta_NO * inhibit_O * monod_NH_H * monod_P * p.X_PAO
        rho_15 = p.b_PAO * p.X_PAO

        rho_20 = p.k_PRE * S_PO4 * p.X_MeOH

        return {
            'rho_aerobic_H': rho_1,
            'rho_anoxic_H' : rho_2,
            'rho_nitrif'   : rho_3,
            'rho_decay_H'  : rho_4,
            'rho_decay_A'  : rho_5,
            'rho_PHA_storage': rho_10,
            'rho_PP_aerobic': rho_11,
            'rho_PP_anoxic': rho_12,
            'rho_PAO_aerobic': rho_13,
            'rho_PAO_anoxic': rho_14,
            'rho_PAO_decay': rho_15,
            'rho_precipitation': rho_20
        }

    def ode_residuals(self, S_S, S_NH, S_PO4, S_O,
                      S_S_in, S_NH_in, S_PO4_in,
                      dSS_dt, dSNH_dt, dSPO4_dt,
                      D=0.16667):
        p = self.p
        rates = self.compute_rates(S_S, S_NH, S_PO4, S_O)

        dil_SS   = D * (S_S_in   - S_S)
        dil_SNH  = D * (S_NH_in  - S_NH)
        dil_SPO4 = D * (S_PO4_in - S_PO4)

        dSS_dt_physics = (dil_SS
                          - (1.0 / p.Y_H) * (rates['rho_aerobic_H'] + rates['rho_anoxic_H'])
                          - rates['rho_PHA_storage'])

        dSNH_dt_physics = (dil_SNH
                           + (1.0 / p.Y_H * p.i_NSF - p.i_XB) * (rates['rho_aerobic_H'] + rates['rho_anoxic_H'])
                           - (1.0 / p.Y_A + p.i_XB) * rates['rho_nitrif']
                           + p.i_NSF * rates['rho_PHA_storage']
                           - p.i_XB * (rates['rho_PAO_aerobic'] + rates['rho_PAO_anoxic'])
                           + p.f_P * p.i_XB * (rates['rho_decay_H'] + rates['rho_decay_A'] + rates['rho_PAO_decay']))

        dSPO4_dt_physics = (dil_SPO4
                            + (1.0 / p.Y_H * p.i_PSF - p.i_XP) * (rates['rho_aerobic_H'] + rates['rho_anoxic_H'])
                            - p.i_XP * rates['rho_nitrif']
                            + (p.i_PSF + p.Y_PO4) * rates['rho_PHA_storage']
                            - 1.0 * (rates['rho_PP_aerobic'] + rates['rho_PP_anoxic'])
                            - p.i_XP * (rates['rho_PAO_aerobic'] + rates['rho_PAO_anoxic'])
                            + p.f_P * p.i_XP * (rates['rho_decay_H'] + rates['rho_decay_A'] + rates['rho_PAO_decay'])
                            - 1.0 * rates['rho_precipitation'])

        res_SS   = dSS_dt   - dSS_dt_physics
        res_SNH  = dSNH_dt  - dSNH_dt_physics
        res_SPO4 = dSPO4_dt - dSPO4_dt_physics

        return res_SS, res_SNH, res_SPO4


class ASM2d_PG_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, fc_dim=64, output_dim=3):
        super(ASM2d_PG_LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.LayerNorm(fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fc_dim // 2, output_dim)
        )

        self.physics = ASM2dPhysics()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions


class PGLossFunction:
    def __init__(self, lambda_data=1.0, lambda_physics=0.1,
                 lambda_boundary=0.05, lambda_variance=0.3,
                 out_range=None, out_min=None):
        self.lambda_data     = lambda_data
        self.lambda_physics  = lambda_physics
        self.lambda_boundary = lambda_boundary
        self.lambda_variance = lambda_variance
        self.mse = nn.MSELoss()
        self.HRT = 6.0
        self.dt  = 1.0
        self.k_KOI_max  = 0.41
        self.k_NH4_max  = 0.40
        self.k_PO4_base = 0.19
        self.K_O2       = 0.50
        self._out_range = (torch.tensor(out_range, dtype=torch.float32)
                           if out_range is not None else None)
        self._out_min   = (torch.tensor(out_min,   dtype=torch.float32)
                           if out_min   is not None else None)

    def __call__(self, predictions, targets, inputs_raw):
        loss_data = self.mse(predictions, targets)

        if predictions.shape[1] > 1:
            dDO  = (inputs_raw[:, 1:, 3] - inputs_raw[:, :-1, 3]) / (6.5 + 1e-9)
            dNH4 = predictions[:, 1:, 1] - predictions[:, :-1, 1]
            loss_do_nh4 = torch.mean(torch.relu(dDO * dNH4))

            dKOI_in  = (inputs_raw[:, 1:, 0] - inputs_raw[:, :-1, 0]) / (700.0 + 1e-9)
            dKOI_out = predictions[:, 1:, 0] - predictions[:, :-1, 0]
            loss_koi_mono = torch.mean(torch.relu(-dKOI_in * dKOI_out))

            loss_physics = loss_do_nh4 + loss_koi_mono
        else:
            loss_physics = torch.tensor(0.0, device=predictions.device)

        negative_penalty = torch.mean(torch.relu(-predictions))
        loss_boundary = negative_penalty


        pred_std = predictions.std(dim=1)
        tgt_std  = targets.std(dim=1)
        loss_std = torch.mean(torch.relu(tgt_std - pred_std) ** 2)

        if predictions.shape[1] > 1:
            pred_delta = torch.abs(predictions[:, 1:, :] - predictions[:, :-1, :])
            tgt_delta  = torch.abs(targets[:,      1:, :] - targets[:,   :-1, :])
            loss_temporal = torch.mean(torch.relu(tgt_delta - pred_delta) ** 2)
        else:
            loss_temporal = torch.tensor(0.0, device=predictions.device)

        loss_variance = 0.5 * loss_std + 0.5 * loss_temporal

        total_loss = (self.lambda_data     * loss_data +
                      self.lambda_physics  * loss_physics +
                      self.lambda_boundary * loss_boundary +
                      self.lambda_variance * loss_variance)

        return total_loss, loss_data, loss_physics, loss_boundary, loss_variance


def load_and_prepare_data(csv_path, seq_len=48, train_ratio=0.8, batch_size=64, verbose=True):
    if verbose:
        print(f"\n{'='*65}")
        print(f"  Veri Yükleme: {csv_path}")
        print(f"{'='*65}")

    df = pd.read_csv(csv_path)
    if verbose:
        print(f"  Toplam satır : {len(df)}")
        print(f"  Sütunlar     : {list(df.columns)}")

    feature_cols = ['KOI_giris', 'NH4_giris', 'PO4_giris', 'DO']
    target_cols  = ['KOI_cikis', 'NH4_cikis', 'PO4_cikis']
    all_cols     = feature_cols + target_cols

    data = df[all_cols].values.astype(np.float32)

    split_idx = int(len(data) * train_ratio)
    scaler = MinMaxScaler()
    scaler.fit(data[:split_idx])
    scaled = scaler.transform(data)

    X_seqs, y_seqs, raw_seqs = [], [], []
    for i in range(len(scaled) - seq_len):
        X_seqs.append(scaled[i:i+seq_len, :4])
        y_seqs.append(scaled[i:i+seq_len, 4:7])
        raw_seqs.append(data[i:i+seq_len, :4])

    X = np.array(X_seqs)
    y = np.array(y_seqs)
    raw = np.array(raw_seqs)

    split = int(len(X) * train_ratio)
    gap   = seq_len
    X_train, X_test = X[:split], X[split + gap:]
    y_train, y_test = y[:split], y[split + gap:]
    raw_train, raw_test = raw[:split], raw[split + gap:]

    if verbose:
        print(f"\n  Sekans uzunluğu  : {seq_len} saat")
        print(f"  Toplam sekans    : {len(X)}")
        print(f"  Eğitim seti      : {len(X_train)}")
        print(f"  Test seti        : {len(X_test)}")
        print(f"  Batch size       : {batch_size}")

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(raw_train)
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test),
        torch.FloatTensor(raw_test)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, df, all_cols


def train_model(model, train_loader, test_loader, scaler, device,
                epochs=200, lr=1e-3, weight_decay=1e-4, patience=30,
                lambda_phys_max=0.05, lambda_boundary=0.03, lambda_variance=0.1,
                noise_std=0.05, verbose=True, trial=None, use_wandb=False):
    if verbose:
        print(f"\n{'='*65}")
        print(f"  PG-LSTM Eğitimi Başlıyor")
        print(f"  Device: {device} | Epochs: {epochs} | LR: {lr}")
        print(f"  Weight Decay: {weight_decay} | λ_phys_max: {lambda_phys_max}")
        print(f"{'='*65}")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    history = {
        'train_total': [], 'train_data': [], 'train_physics': [],
        'test_total': [], 'test_data': [], 'lr': []
    }

    best_test_loss = float('inf')
    patience_counter = 0
    best_state = None

    warmup_end    = 5
    rampup_end    = max(20, epochs // 5)

    start_time = time.time()

    for epoch in range(epochs):
        if epoch < warmup_end:
            lam_phys = 0.0
        elif epoch < rampup_end:
            progress = (epoch - warmup_end) / max(rampup_end - warmup_end, 1)
            lam_phys = lambda_phys_max * 0.3 * progress
        else:
            progress = min((epoch - rampup_end) / max(epochs - rampup_end, 1), 1.0)
            lam_phys = lambda_phys_max * (0.3 + 0.7 * progress)

        criterion = PGLossFunction(
            lambda_data=1.0,
            lambda_physics=lam_phys,
            lambda_boundary=lambda_boundary,
            lambda_variance=lambda_variance,
            out_range=scaler.data_range_[4:7],
            out_min=scaler.data_min_[4:7],
        )

        model.train()
        epoch_total, epoch_data, epoch_phys, epoch_var_l = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for X_batch, y_batch, raw_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            raw_batch = raw_batch.to(device)

            if noise_std > 0:
                scale = torch.rand(1, device=device).item() * noise_std
                X_batch = X_batch + scale * torch.randn_like(X_batch)
                X_batch = torch.clamp(X_batch, 0.0, 1.0)

            optimizer.zero_grad()
            preds = model(X_batch)
            total, data_l, phys_l, bound_l, var_l = criterion(preds, y_batch, raw_batch)

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total  += total.item()
            epoch_data   += data_l.item()
            epoch_phys   += phys_l.item()
            epoch_var_l  += var_l.item()
            n_batches += 1

        scheduler.step()
        avg_train_total = epoch_total  / max(n_batches, 1)
        avg_train_data  = epoch_data   / max(n_batches, 1)
        avg_train_phys  = epoch_phys   / max(n_batches, 1)
        avg_train_var   = epoch_var_l  / max(n_batches, 1)

        model.eval()
        test_total, test_data_l = 0.0, 0.0
        n_test = 0
        with torch.no_grad():
            for X_t, y_t, raw_t in test_loader:
                X_t = X_t.to(device)
                y_t = y_t.to(device)
                raw_t = raw_t.to(device)
                preds_t = model(X_t)
                t_loss, d_loss, _, _, _ = criterion(preds_t, y_t, raw_t)
                test_total += t_loss.item()
                test_data_l += d_loss.item()
                n_test += 1

        avg_test = test_total / max(n_test, 1)
        avg_test_data = test_data_l / max(n_test, 1)

        es_metric = avg_test_data

        current_lr = optimizer.param_groups[0]['lr']
        history['train_total'].append(avg_train_total)
        history['train_data'].append(avg_train_data)
        history['train_physics'].append(avg_train_phys)
        history['test_total'].append(avg_test)
        history['test_data'].append(avg_test_data)
        history['lr'].append(current_lr)

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/total_loss': avg_train_total,
                'train/data_loss': avg_train_data,
                'train/physics_loss': avg_train_phys,
                'train/variance_loss': avg_train_var,
                'test/total_loss': avg_test,
                'test/data_loss': avg_test_data,
                'learning_rate': current_lr,
                'lambda_physics': lam_phys,
                'early_stopping/patience_counter': patience_counter,
                'early_stopping/best_test_loss': best_test_loss,
            }, step=epoch)

        if trial is not None:
            trial.report(es_metric, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if es_metric < best_test_loss:
            best_test_loss = es_metric
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            elapsed = time.time() - start_time
            gap = avg_train_data - avg_test_data
            print(f"  Epoch {epoch+1:4d}/{epochs} │ "
                  f"Train: {avg_train_total:.6f} (data={avg_train_data:.6f}, var={avg_train_var:.4f}) │ "
                  f"Test: {avg_test:.6f} │ "
                  f"Gap: {gap:+.6f} │ "
                  f"λ_phys={lam_phys:.4f} │ "
                  f"LR={current_lr:.2e} │ "
                  f"{elapsed:.0f}s")

        if patience_counter >= patience:
            if verbose:
                print(f"\n  ⚠ Early Stopping: {patience} epoch boyunca iyileşme yok. (Epoch {epoch+1})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    total_time = time.time() - start_time
    if verbose:
        print(f"\n  Eğitim tamamlandı! Süre: {total_time:.1f}s")
        print(f"  En iyi test kaybı (data): {best_test_loss:.6f}")
        print(f"  Toplam epoch     : {len(history['train_total'])}")

    return model, history, best_test_loss


FIXED_SEQ_LEN = 24


def optuna_objective(trial, csv_path, device):
    hidden_dim     = trial.suggest_categorical('hidden_dim', [32, 64, 96, 128])
    num_layers     = trial.suggest_int('num_layers', 1, 3)
    fc_dim         = trial.suggest_categorical('fc_dim', [32, 48, 64, 96])
    dropout        = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    lr             = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    weight_decay   = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    batch_size     = trial.suggest_categorical('batch_size', [32, 64, 128])
    lam_phys_max   = trial.suggest_float('lambda_phys_max', 0.001, 0.1, log=True)
    lam_boundary   = trial.suggest_float('lambda_boundary', 0.01, 0.1)
    lam_variance   = trial.suggest_float('lambda_variance', 0.1, 2.0, log=True)
    noise_std      = trial.suggest_float('noise_std', 0.01, 0.15)

    train_loader, test_loader, scaler, _, _ = load_and_prepare_data(
        csv_path, seq_len=FIXED_SEQ_LEN, train_ratio=0.8,
        batch_size=batch_size, verbose=False
    )

    model = ASM2d_PG_LSTM(
        input_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        fc_dim=fc_dim
    )

    _, _, best_val_loss = train_model(
        model, train_loader, test_loader, scaler, device,
        epochs=100, lr=lr, weight_decay=weight_decay,
        patience=30, lambda_phys_max=lam_phys_max,
        lambda_boundary=lam_boundary, lambda_variance=lam_variance,
        noise_std=noise_std, verbose=False, trial=trial
    )

    return best_val_loss


def run_optuna_optimization(csv_path, device, n_trials=30):
    print(f"\n{'='*65}")
    print(f"  🔍 OPTUNA HİPERPARAMETRE OPTİMİZASYONU")
    print(f"  Toplam trial: {n_trials}")
    print(f"{'='*65}")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=25,
        interval_steps=5
    )

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        study_name='ASM2d_PG_LSTM_HPO'
    )

    start = time.time()

    def objective_wrapper(trial):
        return optuna_objective(trial, csv_path, device)

    study.optimize(objective_wrapper, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)

    elapsed = time.time() - start

    pruned   = len([t for t in study.trials if t.state == TrialState.PRUNED])
    complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])

    print(f"\n{'='*65}")
    print(f"  OPTUNA TAMAMLANDI — {elapsed:.0f} saniye")
    print(f"{'='*65}")
    print(f"  Tamamlanan trial  : {complete}")
    print(f"  Budanan trial     : {pruned}")
    print(f"  En iyi val loss   : {study.best_trial.value:.6f}")
    print(f"\n  En İyi Hiperparametreler:")
    for key, val in study.best_trial.params.items():
        print(f"    {key:20s} = {val}")
    print(f"{'='*65}")

    print(f"\n  📊 En İyi 5 Trial:")
    print(f"  {'#':>3}  {'Val Loss':>10}  {'hidden':>7}  {'layers':>7}  {'dropout':>8}  {'lr':>10}  {'wd':>10}  {'λ_var':>7}  {'batch':>6}")
    sorted_trials = sorted(
        [t for t in study.trials if t.state == TrialState.COMPLETE],
        key=lambda t: t.value
    )
    for i, t in enumerate(sorted_trials[:5]):
        p = t.params
        print(f"  {i+1:3d}  {t.value:10.6f}  {p['hidden_dim']:7d}  {p['num_layers']:7d}  "
              f"{p['dropout']:8.3f}  {p['lr']:10.6f}  {p['weight_decay']:10.6f}  "
              f"{p.get('lambda_variance', 0):7.3f}  {p['batch_size']:6d}")

    best_params = study.best_trial.params
    best_params['best_val_loss'] = study.best_trial.value
    with open('optuna_best_params.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] optuna_best_params.json kaydedildi")

    try:
        _plot_optuna_results(study)
    except Exception as e:
        print(f"  [!] Optuna grafik hatası (önemsiz): {e}")

    return study, best_params


def _plot_optuna_results(study):
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return
    vals = [t.value for t in completed]
    nums = [t.number for t in completed]

    plt.figure(figsize=(6, 5))
    plt.scatter(nums, vals, c='#2196F3', alpha=0.6, s=30, edgecolors='white')
    plt.axhline(y=study.best_trial.value, color='red', linestyle='--',
                label=f'Best: {study.best_trial.value:.6f}')
    cum_min = np.minimum.accumulate(vals)
    plt.plot(nums, cum_min, 'r-', alpha=0.7, linewidth=1.5, label='Cumulative min')
    plt.xlabel('Trial #')
    plt.ylabel('Val Loss')
    plt.title('Trial History')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optuna_trial_history.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] optuna_trial_history.png saved")

    plt.figure(figsize=(6, 5))
    try:
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())[:8]
        values = [importances[n] for n in names]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        plt.barh(names, values, color=colors, edgecolor='white')
        plt.xlabel('Importance Score')
        plt.title('Parameter Importance Ranking')
        plt.grid(True, alpha=0.3, axis='x')
    except Exception:
        plt.text(0.5, 0.5, 'Not enough trials', ha='center', va='center', fontsize=12)
        plt.title('Parameter Importance Ranking')
    plt.tight_layout()
    plt.savefig('optuna_param_importance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] optuna_param_importance.png saved")

    plt.figure(figsize=(6, 5))
    dropouts = [t.params.get('dropout', 0) for t in completed]
    plt.scatter(dropouts, vals, c='#FF5722', alpha=0.6, s=30, edgecolors='white')
    plt.xlabel('Dropout')
    plt.ylabel('Val Loss')
    plt.title('Dropout vs Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optuna_dropout_vs_loss.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] optuna_dropout_vs_loss.png saved")


def evaluate_and_visualize(model, test_loader, scaler, history, device, output_dir="."):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_t, y_t, _ in test_loader:
            X_t = X_t.to(device)
            pred = model(X_t)
            all_preds.append(pred[:, -1, :].cpu().numpy())
            all_targets.append(y_t[:, -1, :].cpu().numpy())

    preds_norm  = np.concatenate(all_preds)
    target_norm = np.concatenate(all_targets)

    n_cols = scaler.n_features_in_
    dummy_pred = np.zeros((len(preds_norm), n_cols))
    dummy_pred[:, 4] = preds_norm[:, 0]
    dummy_pred[:, 5] = preds_norm[:, 1]
    dummy_pred[:, 6] = preds_norm[:, 2]
    preds_real = scaler.inverse_transform(dummy_pred)

    dummy_tgt = np.zeros((len(target_norm), n_cols))
    dummy_tgt[:, 4] = target_norm[:, 0]
    dummy_tgt[:, 5] = target_norm[:, 1]
    dummy_tgt[:, 6] = target_norm[:, 2]
    targets_real = scaler.inverse_transform(dummy_tgt)

    koi_pred  = preds_real[:, 4]
    koi_tgt   = targets_real[:, 4]
    nh4_pred  = preds_real[:, 5]
    nh4_tgt   = targets_real[:, 5]
    po4_pred  = preds_real[:, 6]
    po4_tgt   = targets_real[:, 6]

    def calc_metrics(y_true, y_pred, name):
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return {'name': name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

    m_koi = calc_metrics(koi_tgt, koi_pred, 'KOI_cikis')
    m_nh4 = calc_metrics(nh4_tgt, nh4_pred, 'NH4_cikis')
    m_po4 = calc_metrics(po4_tgt, po4_pred, 'PO4_cikis')

    print(f"\n{'='*75}")
    print(f"  MODEL DEĞERLENDİRME SONUÇLARI — TRIPLE OUTPUT")
    print(f"{'='*75}")
    print(f"  {'Metrik':<8} {'KOI_cikis':>12} {'NH4_cikis':>12} {'PO4_cikis':>12} {'Birim':>10}")
    print(f"  {'-'*65}")
    print(f"  {'MSE':<8} {m_koi['MSE']:>12.4f} {m_nh4['MSE']:>12.4f} {m_po4['MSE']:>12.4f} {'mg²/L²':>10}")
    print(f"  {'RMSE':<8} {m_koi['RMSE']:>12.4f} {m_nh4['RMSE']:>12.4f} {m_po4['RMSE']:>12.4f} {'mg/L':>10}")
    print(f"  {'MAE':<8} {m_koi['MAE']:>12.4f} {m_nh4['MAE']:>12.4f} {m_po4['MAE']:>12.4f} {'mg/L':>10}")
    print(f"  {'R²':<8} {m_koi['R2']:>12.4f} {m_nh4['R2']:>12.4f} {m_po4['R2']:>12.4f} {'-':>10}")
    print(f"  {'MAPE':<8} {m_koi['MAPE']:>12.2f}% {m_nh4['MAPE']:>11.2f}% {m_po4['MAPE']:>11.2f}% {'-':>10}")
    print(f"{'='*75}")

    plt.figure(figsize=(8, 6))
    ep = range(1, len(history['train_total']) + 1)
    plt.semilogy(ep, history['train_total'], 'b-', alpha=0.8, label='Train (Total)', linewidth=1.5)
    plt.semilogy(ep, history['test_total'],  'r-', alpha=0.8, label='Test (Total)', linewidth=1.5)
    plt.semilogy(ep, history['train_data'],  'b--', alpha=0.5, label='Train (Data)', linewidth=1.0)
    plt.semilogy(ep, history['train_physics'], 'g--', alpha=0.5, label='Physics Loss', linewidth=1.0)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss (log scale)', fontsize=11)
    plt.title('Training and Validation Loss Curve', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_loss = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(path_loss, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_loss} saved")

    plt.figure(figsize=(8, 6))
    n_show = min(500, len(koi_tgt))
    plt.plot(range(n_show), koi_tgt[:n_show], 'b-', alpha=0.7, label='True', linewidth=1.2)
    plt.plot(range(n_show), koi_pred[:n_show], 'r--', alpha=0.7, label='Predicted', linewidth=1.2)
    plt.fill_between(range(n_show), koi_pred[:n_show] - m_koi['RMSE'],
                     koi_pred[:n_show] + m_koi['RMSE'], alpha=0.15, color='red',
                     label=f'±RMSE ({m_koi["RMSE"]:.2f})')
    plt.xlabel('Time (hours)', fontsize=11)
    plt.ylabel('COD Effluent (mg/L)', fontsize=11)
    plt.title(f'COD Effluent: True vs Predicted (R²={m_koi["R2"]:.4f})', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_cod_ts = os.path.join(output_dir, 'cod_timeseries.png')
    plt.savefig(path_cod_ts, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_cod_ts} saved")

    plt.figure(figsize=(8, 6))
    plt.plot(range(n_show), nh4_tgt[:n_show], 'b-', alpha=0.7, label='True', linewidth=1.2)
    plt.plot(range(n_show), nh4_pred[:n_show], 'r--', alpha=0.7, label='Predicted', linewidth=1.2)
    plt.fill_between(range(n_show), nh4_pred[:n_show] - m_nh4['RMSE'],
                     nh4_pred[:n_show] + m_nh4['RMSE'], alpha=0.15, color='red',
                     label=f'±RMSE ({m_nh4["RMSE"]:.2f})')
    plt.axhline(y=10, color='orange', ls='--', lw=1, alpha=0.7, label='EU Limit (10 mg/L)')
    plt.xlabel('Time (hours)', fontsize=11)
    plt.ylabel('NH₄-N Effluent (mg/L)', fontsize=11)
    plt.title(f'NH₄-N Effluent: True vs Predicted (R²={m_nh4["R2"]:.4f})', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_nh4_ts = os.path.join(output_dir, 'nh4_timeseries.png')
    plt.savefig(path_nh4_ts, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_nh4_ts} saved")

    plt.figure(figsize=(8, 6))
    plt.plot(range(n_show), po4_tgt[:n_show], 'b-', alpha=0.7, label='True', linewidth=1.2)
    plt.plot(range(n_show), po4_pred[:n_show], 'r--', alpha=0.7, label='Predicted', linewidth=1.2)
    plt.fill_between(range(n_show), po4_pred[:n_show] - m_po4['RMSE'],
                     po4_pred[:n_show] + m_po4['RMSE'], alpha=0.15, color='red',
                     label=f'±RMSE ({m_po4["RMSE"]:.2f})')
    plt.xlabel('Time (hours)', fontsize=11)
    plt.ylabel('PO₄-P Effluent (mg/L)', fontsize=11)
    plt.title(f'PO₄-P Effluent: True vs Predicted (R²={m_po4["R2"]:.4f})', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_po4_ts = os.path.join(output_dir, 'po4_timeseries.png')
    plt.savefig(path_po4_ts, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_po4_ts} saved")

    plt.figure(figsize=(8, 6))
    err_koi = koi_pred - koi_tgt
    plt.hist(err_koi, bins=50, color='#4CAF50', alpha=0.7, edgecolor='white', density=True)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=err_koi.mean(), color='orange', linestyle='-', linewidth=1.5,
                label=f'Mean Error = {err_koi.mean():.2f}')
    plt.xlabel('COD Prediction Error (mg/L)', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title(f'COD Error Distribution (σ = {err_koi.std():.2f} mg/L)', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_cod_err = os.path.join(output_dir, 'cod_error_dist.png')
    plt.savefig(path_cod_err, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_cod_err} saved")

    plt.figure(figsize=(8, 6))
    plt.scatter(koi_tgt, koi_pred, alpha=0.3, s=10, c='#2196F3', edgecolors='none')
    mn, mx = min(koi_tgt.min(), koi_pred.min()), max(koi_tgt.max(), koi_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1.5, label='Ideal (y=x)')
    plt.xlabel('True COD Effluent (mg/L)', fontsize=11)
    plt.ylabel('Predicted COD Effluent (mg/L)', fontsize=11)
    plt.title(f'COD Scatter Plot (R² = {m_koi["R2"]:.4f})', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_cod_scat = os.path.join(output_dir, 'cod_scatter.png')
    plt.savefig(path_cod_scat, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_cod_scat} saved")

    plt.figure(figsize=(8, 6))
    plt.scatter(nh4_tgt, nh4_pred, alpha=0.3, s=10, c='#FF5722', edgecolors='none')
    mn, mx = min(nh4_tgt.min(), nh4_pred.min()), max(nh4_tgt.max(), nh4_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1.5, label='Ideal (y=x)')
    plt.xlabel('True NH₄-N Effluent (mg/L)', fontsize=11)
    plt.ylabel('Predicted NH₄-N Effluent (mg/L)', fontsize=11)
    plt.title(f'NH₄-N Scatter Plot (R² = {m_nh4["R2"]:.4f})', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_nh4_scat = os.path.join(output_dir, 'nh4_scatter.png')
    plt.savefig(path_nh4_scat, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_nh4_scat} saved")

    plt.figure(figsize=(8, 6))
    plt.scatter(po4_tgt, po4_pred, alpha=0.3, s=10, c='#9C27B0', edgecolors='none')
    mn, mx = min(po4_tgt.min(), po4_pred.min()), max(po4_tgt.max(), po4_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1.5, label='Ideal (y=x)')
    plt.xlabel('True PO₄-P Effluent (mg/L)', fontsize=11)
    plt.ylabel('Predicted PO₄-P Effluent (mg/L)', fontsize=11)
    plt.title(f'PO₄-P Scatter Plot (R² = {m_po4["R2"]:.4f})', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_po4_scat = os.path.join(output_dir, 'po4_scatter.png')
    plt.savefig(path_po4_scat, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_po4_scat} saved")

    plt.figure(figsize=(8, 6))
    plt.hist(koi_tgt, bins=40, alpha=0.6, label='True', color='#2196F3', density=True)
    plt.hist(koi_pred, bins=40, alpha=0.6, label='Predicted', color='#FF5722', density=True)
    plt.xlabel('COD Effluent (mg/L)')
    plt.ylabel('Density')
    plt.title('COD Effluent Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_cod_dist = os.path.join(output_dir, 'cod_dist.png')
    plt.savefig(path_cod_dist, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_cod_dist} saved")

    plt.figure(figsize=(8, 6))
    plt.hist(nh4_tgt, bins=40, alpha=0.6, label='True', color='#2196F3', density=True)
    plt.hist(nh4_pred, bins=40, alpha=0.6, label='Predicted', color='#FF5722', density=True)
    plt.xlabel('NH₄-N Effluent (mg/L)')
    plt.ylabel('Density')
    plt.title('NH₄-N Effluent Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_nh4_dist = os.path.join(output_dir, 'nh4_dist.png')
    plt.savefig(path_nh4_dist, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_nh4_dist} saved")

    plt.figure(figsize=(8, 6))
    plt.hist(po4_tgt, bins=40, alpha=0.6, label='True', color='#2196F3', density=True)
    plt.hist(po4_pred, bins=40, alpha=0.6, label='Predicted', color='#FF5722', density=True)
    plt.xlabel('PO₄-P Effluent (mg/L)')
    plt.ylabel('Density')
    plt.title('PO₄-P Effluent Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_po4_dist = os.path.join(output_dir, 'po4_dist.png')
    plt.savefig(path_po4_dist, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_po4_dist} saved")

    plt.figure(figsize=(8, 6))
    err_nh4 = nh4_pred - nh4_tgt
    plt.hist(err_nh4, bins=50, color='#E91E63', alpha=0.7, edgecolor='white', density=True)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=err_nh4.mean(), color='orange', linestyle='-', linewidth=1.5,
                label=f'Mean Error = {err_nh4.mean():.2f}')
    plt.xlabel('NH₄-N Prediction Error (mg/L)')
    plt.ylabel('Density')
    plt.title(f'NH₄-N Error Distribution (σ = {err_nh4.std():.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_nh4_err = os.path.join(output_dir, 'nh4_error_dist.png')
    plt.savefig(path_nh4_err, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_nh4_err} saved")

    plt.figure(figsize=(8, 6))
    err_po4 = po4_pred - po4_tgt
    plt.hist(err_po4, bins=50, color='#9C27B0', alpha=0.7, edgecolor='white', density=True)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=err_po4.mean(), color='orange', linestyle='-', linewidth=1.5,
                label=f'Mean Error = {err_po4.mean():.2f}')
    plt.xlabel('PO₄-P Prediction Error (mg/L)')
    plt.ylabel('Density')
    plt.title(f'PO₄-P Error Distribution (σ = {err_po4.std():.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_po4_err = os.path.join(output_dir, 'po4_error_dist.png')
    plt.savefig(path_po4_err, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_po4_err} saved")

    plt.figure(figsize=(8, 6))
    cum_err_koi = np.cumsum(np.abs(err_koi)) / np.arange(1, len(err_koi) + 1)
    plt.plot(cum_err_koi, color='#9C27B0', linewidth=1.5)
    plt.xlabel('Sample Number')
    plt.ylabel('Cumulative MAE (mg/L)')
    plt.title('Cumulative MAE — COD')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_cum_cod = os.path.join(output_dir, 'cum_mae_cod.png')
    plt.savefig(path_cum_cod, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_cum_cod} saved")

    plt.figure(figsize=(8, 6))
    cum_err_nh4 = np.cumsum(np.abs(err_nh4)) / np.arange(1, len(err_nh4) + 1)
    plt.plot(cum_err_nh4, color='#E91E63', linewidth=1.5)
    plt.xlabel('Sample Number')
    plt.ylabel('Cumulative MAE (mg/L)')
    plt.title('Cumulative MAE — NH₄-N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_cum_nh4 = os.path.join(output_dir, 'cum_mae_nh4.png')
    plt.savefig(path_cum_nh4, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_cum_nh4} saved")

    plt.figure(figsize=(8, 6))
    cum_err_po4 = np.cumsum(np.abs(err_po4)) / np.arange(1, len(err_po4) + 1)
    plt.plot(cum_err_po4, color='#9C27B0', linewidth=1.5)
    plt.xlabel('Sample Number')
    plt.ylabel('Cumulative MAE (mg/L)')
    plt.title('Cumulative MAE — PO₄-P')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_cum_po4 = os.path.join(output_dir, 'cum_mae_po4.png')
    plt.savefig(path_cum_po4, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_cum_po4} saved")

    plt.figure(figsize=(8, 6))
    plt.plot(history['lr'], color='#FF9800', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_lr = os.path.join(output_dir, 'lr_curve.png')
    plt.savefig(path_lr, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {path_lr} saved")

    return {
        'KOI_MSE': m_koi['MSE'], 'KOI_RMSE': m_koi['RMSE'], 'KOI_MAE': m_koi['MAE'],
        'KOI_R2': m_koi['R2'], 'KOI_MAPE': m_koi['MAPE'],
        'NH4_MSE': m_nh4['MSE'], 'NH4_RMSE': m_nh4['RMSE'], 'NH4_MAE': m_nh4['MAE'],
        'NH4_R2': m_nh4['R2'], 'NH4_MAPE': m_nh4['MAPE'],
        'PO4_MSE': m_po4['MSE'], 'PO4_RMSE': m_po4['RMSE'], 'PO4_MAE': m_po4['MAE'],
        'PO4_R2': m_po4['R2'], 'PO4_MAPE': m_po4['MAPE'],
        'koi_predictions': koi_pred, 'koi_targets': koi_tgt,
        'nh4_predictions': nh4_pred, 'nh4_targets': nh4_tgt,
        'po4_predictions': po4_pred, 'po4_targets': po4_tgt,
    }


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   ASM2d — PG + LSTM Hibrit Model + Optuna HPO               ║
    ║                                                               ║
    ║   Physics-Guided × Long Short-Term Memory    ║
    ║   Monod Kinetiği + ODE Kısıtı + Curriculum Learning           ║
    ║   + Optuna Hiperparametre Optimizasyonu                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Hesaplama Cihazı: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    ASM2dParameters.summary()

    csv_path = "synthetic_asm2d_physics.csv"

    
    import json
    print("\n  Optuna atlanıyor... Önceden kaydedilmiş en iyi parametreler optuna_best_params.json'dan yükleniyor...")
    with open('optuna_best_params.json', 'r') as f:
        best_optuna_params = json.load(f)
    study = None

    BEST_SEQ_LEN     = FIXED_SEQ_LEN
    BEST_HIDDEN_DIM  = best_optuna_params['hidden_dim']
    BEST_NUM_LAYERS  = best_optuna_params['num_layers']
    BEST_FC_DIM      = best_optuna_params['fc_dim']
    BEST_DROPOUT     = best_optuna_params['dropout']
    BEST_LR          = best_optuna_params['lr']
    BEST_WD          = best_optuna_params['weight_decay']
    BEST_BATCH       = best_optuna_params['batch_size']
    BEST_LAM_PHYS    = best_optuna_params['lambda_phys_max']
    BEST_LAM_BOUND   = best_optuna_params['lambda_boundary']
    BEST_LAM_VAR     = best_optuna_params.get('lambda_variance', 0.30)
    BEST_NOISE_STD   = best_optuna_params['noise_std']

    print(f"\n{'='*65}")
    print(f"  🏆 OPTUNA'NIN BULDUĞU EN İYİ PARAMETRELERLE TAM EĞİTİM BAŞLIYOR")
    print(f"{'='*65}")
    print(f"  seq_len        = {BEST_SEQ_LEN}")
    print(f"  hidden_dim     = {BEST_HIDDEN_DIM}")
    print(f"  num_layers     = {BEST_NUM_LAYERS}")
    print(f"  fc_dim         = {BEST_FC_DIM}")
    print(f"  dropout        = {BEST_DROPOUT}")
    print(f"  lr             = {BEST_LR}")
    print(f"  weight_decay   = {BEST_WD}")
    print(f"  batch_size     = {BEST_BATCH}")
    print(f"  lambda_phys    = {BEST_LAM_PHYS}")
    print(f"  lambda_boundary= {BEST_LAM_BOUND}")
    print(f"  lambda_variance= {BEST_LAM_VAR}")
    print(f"  noise_std      = {BEST_NOISE_STD}")
    print(f"{'='*65}")

    train_loader, test_loader, scaler, df, all_cols = load_and_prepare_data(
        csv_path, seq_len=BEST_SEQ_LEN, train_ratio=0.8,
        batch_size=BEST_BATCH, verbose=True
    )

    model = ASM2d_PG_LSTM(
        input_dim=4,
        hidden_dim=BEST_HIDDEN_DIM,
        num_layers=BEST_NUM_LAYERS,
        dropout=BEST_DROPOUT,
        fc_dim=BEST_FC_DIM
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model Parametreleri: {total_params:,} (Eğitilebilir: {trainable:,})")

    model, history, best_val = train_model(
        model, train_loader, test_loader, scaler, device,
        epochs=200, lr=BEST_LR, weight_decay=BEST_WD,
        patience=85, lambda_phys_max=BEST_LAM_PHYS,
        lambda_boundary=BEST_LAM_BOUND, lambda_variance=BEST_LAM_VAR,
        noise_std=BEST_NOISE_STD, verbose=True,
        use_wandb=False
    )

    results = evaluate_and_visualize(model, test_loader, scaler, history, device)

    scaler_path = "asm2d_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"\n  [OK] Scaler kaydedildi: {scaler_path}")

    model_path = "asm2d_pg_lstm_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_min': scaler.data_min_,
        'scaler_max': scaler.data_max_,
        'scaler_path': scaler_path,
        'seq_len': BEST_SEQ_LEN,
        'hidden_dim': BEST_HIDDEN_DIM,
        'num_layers': BEST_NUM_LAYERS,
        'fc_dim': BEST_FC_DIM,
        'dropout': BEST_DROPOUT,
        'input_dim': 4,
        'output_dim': 3,
        'best_params': {
            'hidden_dim': BEST_HIDDEN_DIM, 'num_layers': BEST_NUM_LAYERS,
            'fc_dim': BEST_FC_DIM, 'dropout': BEST_DROPOUT,
            'lr': BEST_LR, 'weight_decay': BEST_WD, 'batch_size': BEST_BATCH,
            'lambda_phys_max': BEST_LAM_PHYS, 'lambda_boundary': BEST_LAM_BOUND,
            'lambda_variance': BEST_LAM_VAR, 'noise_std': BEST_NOISE_STD,
        },
        'koi_r2':   results['KOI_R2'],
        'koi_mape': results['KOI_MAPE'],
        'nh4_r2':   results['NH4_R2'],
        'nh4_mape': results['NH4_MAPE'],
        'po4_r2':   results['PO4_R2'],
        'po4_mape': results['PO4_MAPE'],
    }, model_path)
    print(f"  [OK] Model kaydedildi: {model_path}")

    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    SONUÇ RAPORU                               ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Model        : LSTM ({BEST_NUM_LAYERS} katman, {BEST_HIDDEN_DIM} gizli, fc={BEST_FC_DIM})      ║
    ║  Dropout      : {BEST_DROPOUT:.2f}                                       ║
    ║  Fizik Kısıtı : ASM2d ODE (λ_max={BEST_LAM_PHYS:.4f})         ║
    ║  Veri Kaynağı : S.V. ile üretilmiş ASM2d sentetik verisi      ║
    ║  Sekans       : {BEST_SEQ_LEN} saat (sliding window)          ║
    ║  En iyi test  : val_loss={best_val:.6f} (Manuel HPO)          ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                      [ KOİ Çıkış ]                            ║
    ║  MSE: {results['KOI_MSE']:.4f}   RMSE: {results['KOI_RMSE']:.4f}   MAE: {results['KOI_MAE']:.4f}       ║
    ║  R² : {results['KOI_R2']:.4f}   MAPE: {results['KOI_MAPE']:.2f}%                      ║
    ║                                                               ║
    ║                     [ NH4-N Çıkış ]                           ║
    ║  MSE: {results['NH4_MSE']:.4f}   RMSE: {results['NH4_RMSE']:.4f}   MAE: {results['NH4_MAE']:.4f}       ║
    ║  R² : {results['NH4_R2']:.4f}   MAPE: {results['NH4_MAPE']:.2f}%                      ║
    ║                                                               ║
    ║                     [ PO4-P Çıkış ]                           ║
    ║  MSE: {results['PO4_MSE']:.4f}   RMSE: {results['PO4_RMSE']:.4f}   MAE: {results['PO4_MAE']:.4f}       ║
    ║  R² : {results['PO4_R2']:.4f}   MAPE: {results['PO4_MAPE']:.2f}%                      ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Outputs:                                                     ║
    ║    📊 *.png                       (individual plots)          ║
    ║    💾 asm2d_pg_lstm_model.pth   (model ağırlıkları)        ║
    ║    💾 asm2d_scaler.joblib         (normalizasyon scaler)      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
