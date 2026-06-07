
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os, json, joblib, time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asm2d_pg_lstm import load_and_prepare_data

class ASM2d_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, fc_dim=64, output_dim=3):
        super(ASM2d_GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.LayerNorm(fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2),
            nn.GELU(),
            nn.Linear(fc_dim // 2, output_dim)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)

def train_gru():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GRU egitimi basliyor... Cihaz: {device}")

    params_path = "../optuna_best_params.json"
    if not os.path.exists(params_path): params_path = "optuna_best_params.json"
    with open(params_path, 'r') as f:
        best_params = json.load(f)

    csv_path = "../synthetic_asm2d_physics.csv"
    if not os.path.exists(csv_path): csv_path = "synthetic_asm2d_physics.csv"

    train_loader, test_loader, scaler, _, _ = load_and_prepare_data(
        csv_path, seq_len=48,
        batch_size=best_params['batch_size'], verbose=False
    )

    model = ASM2d_GRU(
        input_dim=4,
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        fc_dim=best_params['fc_dim']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    epochs = 200
    patience = 50
    best_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y, _ in test_loader:
                X, y = X.to(device), y.to(device)
                test_loss += criterion(model(X), y).item()
        
        avg_test = test_loss / len(test_loader)
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Test Loss: {avg_test:.6f}")

        if avg_test < best_loss:
            best_loss = avg_test
            counter = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "gru_model.pth"))
        else:
            counter += 1
            if counter >= patience: break

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "gru_model.pth"), weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y, _ in test_loader:
            preds = model(X.to(device))
            all_preds.append(preds[:, -1, :].cpu().numpy())
            all_targets.append(y[:, -1, :].cpu().numpy())
    
    preds_norm = np.concatenate(all_preds)
    targets_norm = np.concatenate(all_targets)

    dummy_pred = np.zeros((len(preds_norm), scaler.n_features_in_))
    dummy_pred[:, 4:7] = preds_norm
    preds_real = scaler.inverse_transform(dummy_pred)[:, 4:7]

    dummy_tgt = np.zeros((len(targets_norm), scaler.n_features_in_))
    dummy_tgt[:, 4:7] = targets_norm
    targets_real = scaler.inverse_transform(dummy_tgt)[:, 4:7]

    metrics = {}
    names = ['KOI', 'NH4', 'PO4']
    for i, name in enumerate(names):
        r2 = r2_score(targets_real[:, i], preds_real[:, i])
        rmse = np.sqrt(mean_squared_error(targets_real[:, i], preds_real[:, i]))
        metrics[name] = {'R2': r2, 'RMSE': rmse}
        print(f"GRU  {name:3s} -> R2: {r2:.4f} | RMSE: {rmse:.4f}")

    with open(os.path.join(os.path.dirname(__file__), "gru_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train_gru()
