
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os, json, joblib, time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asm2d_pg_lstm import ASM2d_PG_LSTM, load_and_prepare_data

def train_vanilla():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Vanilla LSTM egitimi basliyor... Cihaz: {device}")

    params_path = "optuna_best_params.json"
    if not os.path.exists(params_path):
        params_path = "../optuna_best_params.json"
        
    with open(params_path, 'r') as f:
        best_params = json.load(f)

    csv_path = "synthetic_asm2d_physics.csv"
    if not os.path.exists(csv_path):
         csv_path = "../synthetic_asm2d_physics.csv"

    train_loader, test_loader, scaler, _, _ = load_and_prepare_data(
        csv_path, 
        seq_len=48,
        batch_size=best_params['batch_size'], 
        verbose=True
    )

    model = ASM2d_PG_LSTM(
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
    history = {'train': [], 'test': []}

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        avg_train = train_loss / len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y, _ in test_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                test_loss += criterion(preds, y).item()
        
        avg_test = test_loss / len(test_loader)
        history['train'].append(avg_train)
        history['test'].append(avg_test)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train:.6f} | Test: {avg_test:.6f}")

        if avg_test < best_loss:
            best_loss = avg_test
            counter = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "vanilla_lstm_model.pth"))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Egitim tamamlandi. Sure: {time.time()-start_time:.1f}s")

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "vanilla_lstm_model.pth"), weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y, _ in test_loader:
            X = X.to(device)
            preds = model(X)
            all_preds.append(preds[:, -1, :].cpu().numpy())
            all_targets.append(y[:, -1, :].cpu().numpy())
    
    preds_norm = np.concatenate(all_preds)
    targets_norm = np.concatenate(all_targets)

    n_cols = scaler.n_features_in_
    dummy_pred = np.zeros((len(preds_norm), n_cols))
    dummy_pred[:, 4:7] = preds_norm
    preds_real = scaler.inverse_transform(dummy_pred)[:, 4:7]

    dummy_tgt = np.zeros((len(targets_norm), n_cols))
    dummy_tgt[:, 4:7] = targets_norm
    targets_real = scaler.inverse_transform(dummy_tgt)[:, 4:7]

    metrics = {}
    names = ['KOI', 'NH4', 'PO4']
    print("\nVanilla LSTM Sonuclari:")
    for i, name in enumerate(names):
        r2 = r2_score(targets_real[:, i], preds_real[:, i])
        rmse = np.sqrt(mean_squared_error(targets_real[:, i], preds_real[:, i]))
        metrics[name] = {'R2': r2, 'RMSE': rmse}
        print(f"  {name:3s} -> R2: {r2:.4f} | RMSE: {rmse:.4f}")

    with open(os.path.join(os.path.dirname(__file__), "vanilla_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    train_vanilla()
