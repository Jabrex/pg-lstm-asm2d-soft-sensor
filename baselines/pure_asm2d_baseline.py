
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_pure_ode():
    print("Pure ASM2d ODE / Persistence Baseline degerlendirmesi basliyor...")

    csv_path = os.path.join(os.path.dirname(__file__), "..", "synthetic_asm2d_physics.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "synthetic_asm2d_physics.csv")

    df = pd.read_csv(csv_path)

    split = int(len(df) * 0.8)
    df_test = df.iloc[split:].reset_index(drop=True)

    targets = {
        'KOI': df_test['KOI_cikis'].values,
        'NH4': df_test['NH4_cikis'].values,
        'PO4': df_test['PO4_cikis'].values,
    }

    print("\n[1] Persistence (Naive) Baseline:")
    persistence_metrics = {}
    for name, y in targets.items():
        y_pred = y[:-1]
        y_true = y[1:]
        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        persistence_metrics[name] = {'R2': r2, 'RMSE': rmse}
        print(f"  {name:3s} -> R2: {r2:.4f} | RMSE: {rmse:.4f}")

    print("\n[2] ODE Steady-State Approximation Baseline:")
    euler_metrics = {}

    koi_in  = df_test['KOI_giris'].values
    nh4_in  = df_test['NH4_giris'].values
    po4_in  = df_test['PO4_giris'].values
    do_vals = df_test['DO'].values

    eta_nit = do_vals / (0.50 + do_vals)
    eta_het = do_vals / (0.20 + do_vals) * 0.824

    koi_pred = koi_in * (1.0 - eta_het) + 30.0
    nh4_pred = nh4_in * (1.0 - eta_nit * 0.889)
    po4_pred = po4_in * (1.0 - 0.949)

    pred_map = {'KOI': koi_pred, 'NH4': nh4_pred, 'PO4': po4_pred}
    for name, y in targets.items():
        y_pred = pred_map[name]
        r2   = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        euler_metrics[name] = {'R2': r2, 'RMSE': rmse}
        print(f"  {name:3s} -> R2: {r2:.4f} | RMSE: {rmse:.4f}")
    print("  [NOT: Gercek veri geldiginde bu degerler gercekci ODE hata payini gosterecek]")

    results = {
        'persistence': persistence_metrics,
        'euler_linearization': euler_metrics
    }
    out_path = os.path.join(os.path.dirname(__file__), "pure_ode_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSonuclar kaydedildi: {out_path}")
    return results


if __name__ == "__main__":
    evaluate_pure_ode()
