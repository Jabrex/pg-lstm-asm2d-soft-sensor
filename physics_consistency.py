import numpy as np
import torch
from asm2d_pg_lstm import ASM2dPhysics


def compute_ode_residual_norm(preds_real, raw_inputs, dt=1.0, D=0.16667, reduce='mean_l2'):
    if len(preds_real) < 2:
        raise ValueError("En az 2 zaman adımı gerekli (finite difference için)")

    physics = ASM2dPhysics()

    S_S   = torch.as_tensor(preds_real[:-1, 0], dtype=torch.float32)
    S_NH  = torch.as_tensor(preds_real[:-1, 1], dtype=torch.float32)
    S_PO4 = torch.as_tensor(preds_real[:-1, 2], dtype=torch.float32)

    S_S_in   = torch.as_tensor(raw_inputs[:-1, 0], dtype=torch.float32)
    S_NH_in  = torch.as_tensor(raw_inputs[:-1, 1], dtype=torch.float32)
    S_PO4_in = torch.as_tensor(raw_inputs[:-1, 2], dtype=torch.float32)
    S_O      = torch.as_tensor(raw_inputs[:-1, 3], dtype=torch.float32)

    dSS_dt   = torch.as_tensor((preds_real[1:, 0] - preds_real[:-1, 0]) / dt, dtype=torch.float32)
    dSNH_dt  = torch.as_tensor((preds_real[1:, 1] - preds_real[:-1, 1]) / dt, dtype=torch.float32)
    dSPO4_dt = torch.as_tensor((preds_real[1:, 2] - preds_real[:-1, 2]) / dt, dtype=torch.float32)

    res_SS, res_SNH, res_SPO4 = physics.ode_residuals(
        S_S, S_NH, S_PO4, S_O,
        S_S_in, S_NH_in, S_PO4_in,
        dSS_dt, dSNH_dt, dSPO4_dt,
        D=D
    )

    res_SS   = res_SS.cpu().numpy()
    res_SNH  = res_SNH.cpu().numpy()
    res_SPO4 = res_SPO4.cpu().numpy()

    if reduce == 'per_state':
        return {
            'SS':   float(np.sqrt(np.mean(res_SS   ** 2))),
            'SNH':  float(np.sqrt(np.mean(res_SNH  ** 2))),
            'SPO4': float(np.sqrt(np.mean(res_SPO4 ** 2))),
        }
    elif reduce == 'rms':
        all_res = np.concatenate([res_SS, res_SNH, res_SPO4])
        return float(np.sqrt(np.mean(all_res ** 2)))
    elif reduce == 'mean_l2':
        rms_per_state = [
            np.sqrt(np.mean(res_SS   ** 2)),
            np.sqrt(np.mean(res_SNH  ** 2)),
            np.sqrt(np.mean(res_SPO4 ** 2)),
        ]
        return float(np.mean(rms_per_state))
    else:
        raise ValueError(f"Bilinmeyen reduce: {reduce}")


def compute_normalized_residual(preds_real, raw_inputs, dt=1.0, D=0.16667):
    raw = compute_ode_residual_norm(preds_real, raw_inputs, dt=dt, D=D, reduce='per_state')
    scales = {
        'SS':   1.0,
        'SNH':  0.1,
        'SPO4': 0.1,
    }
    return {f"{k}_norm": raw[k] / scales[k] for k in scales}


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("synthetic_asm2d_physics.csv")
    preds = df[['KOI_cikis', 'NH4_cikis', 'PO4_cikis']].values[:1000]
    raw   = df[['KOI_giris', 'NH4_giris', 'PO4_giris', 'DO']].values[:1000]
    rms = compute_ode_residual_norm(preds, raw, dt=1.0, reduce='per_state')
    print("Per-state ODE residual RMS (gerçek veri üzerinde):")
    for k, v in rms.items():
        print(f"  {k}: {v:.6f}")
    print(f"  mean_l2: {compute_ode_residual_norm(preds, raw, reduce='mean_l2'):.6f}")
