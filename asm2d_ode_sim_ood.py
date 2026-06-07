
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

OOD_SEED = 7
PERT_FRAC = 0.15
INFL_FRAC = 0.15

np.random.seed(OOD_SEED)
_pert_rng = np.random.default_rng(OOD_SEED + 1000)


def _pf():
    return float(1.0 + _pert_rng.uniform(-PERT_FRAC, PERT_FRAC))


def _if():
    return float(1.0 + _pert_rng.uniform(-INFL_FRAC, INFL_FRAC))


PF = {
    'k_hyd': _pf(), 'K_X': _pf(),
    'mu_H': _pf(), 'q_fe': _pf(), 'b_H': _pf(),
    'K_F': _pf(), 'K_A_H': _pf(), 'K_fe': _pf(),
    'q_PHA': _pf(), 'q_PP': _pf(), 'mu_PAO': _pf(),
    'b_PAO': _pf(), 'b_PP': _pf(), 'b_PHA': _pf(),
    'K_A_PAO': _pf(), 'K_PS': _pf(), 'K_MAX': _pf(),
    'mu_AUT': _pf(), 'b_AUT': _pf(), 'K_NH4_AUT': _pf(), 'K_O2_AUT': _pf(),
    'k_PRE': _pf(), 'k_RED': _pf(),
}
IF = {
    'Q_avg': _if(), 'KOI_mean': _if(), 'NH4_mean': _if(), 'PO4_mean': _if(),
    'KOI_amp': _if(), 'NH4_amp': _if(), 'PO4_amp': _if(),
}
PHASE_SHIFT = float(_pert_rng.uniform(-0.6, 0.6))

warnings.filterwarnings('ignore')

WARMUP_H = 720
SIM_H = 8760
N_TOTAL = WARMUP_H + SIM_H

STATE_NAMES = [
    'SO2', 'SF', 'SA', 'SNH4', 'SNO3', 'SPO4', 'SALK', 'SN2',
    'XI', 'XS', 'XH', 'XPAO', 'XPP', 'XPHA', 'XAUT', 'XTSS', 'XMeOH', 'XMeP'
]
S_IDX = {name: i for i, name in enumerate(STATE_NAMES)}
N_STATES = 18
N_ZONES = 3
N_TOTAL_STATES = N_STATES * N_ZONES
SOLUBLE_IDX = list(range(0, 8))
PARTICULATE_IDX = list(range(8, 18))

print("=" * 72)
print("  ASM2d A2O — BAGIMSIZ (OOD) SENARYO URETICISI")
print(f"  Seed={OOD_SEED}  |  kinetik perturbasyon = +-{PERT_FRAC*100:.0f}%")
print(f"  Konfigurasyon : 3 bolge (AN+AX+AE), {N_TOTAL_STATES} ODE state")
print("=" * 72)

V_AN = 1500.0
V_AX = 1500.0
V_AE = 3000.0
V_TOTAL = V_AN + V_AX + V_AE

Q_avg = 1000.0 * IF['Q_avg']
SRT_d = 15.0
SRT_h = SRT_d * 24.0

S_I = 30.0

Q_IR_factor = 3.0
Q_RAS_factor = 1.0

SNO3_IN_BASE = 0.40
SN2_IN_BASE  = 18.0
SALK_IN_BASE = 7.2
XMeOH_IN_BASE = 4.0

kLa_AE      = 22.0
DO_sat      = 8.5
DO_setpoint_AE = 2.5

solids_capture = 0.985
EPS = 1e-9

fSI   = 0.0
YH    = 0.625
fXI   = 0.10
YPAO  = 0.625
YPHA  = 0.20
YPO4  = 0.40
YA    = 0.24
fMeOH = -3.45
fMeP  = 4.86
iNO3_N2 = 2.857

iN_SF  = 0.03; iN_SI = 0.01; iN_XI = 0.02; iN_XS = 0.04; iN_BM = 0.07
iP_SF  = 0.01; iP_SI = 0.00; iP_XI = 0.01; iP_XS = 0.01; iP_BM = 0.02

iTSS_XI   = 0.75; iTSS_XS = 0.75; iTSS_XPHA = 0.60
iTSS_BM   = 0.90; iTSS_XPP = 3.23

k_hyd = (3.0/24.0) * PF['k_hyd']; K_X = 0.10 * PF['K_X']
eta_NO_hyd = 0.60; eta_fe_hyd = 0.40
K_O2_hyd = 0.20; K_NO3_hyd = 0.50

mu_H = (6.0/24.0) * PF['mu_H']; eta_NO_H = 0.80; q_fe = (3.0/24.0) * PF['q_fe']
K_F = 4.0 * PF['K_F']; K_A_H = 4.0 * PF['K_A_H']; b_H = (0.40/24.0) * PF['b_H']
K_fe = 4.0 * PF['K_fe']
K_O2_H = 0.20; K_NO3_H = 0.50; K_NH4_H = 0.05; K_P_H = 0.01; K_ALK_H = 0.10

q_PHA = (3.0/24.0) * PF['q_PHA']; q_PP = (1.5/24.0) * PF['q_PP']
K_PP = 0.01; K_MAX = 0.34 * PF['K_MAX']; K_iPP = 0.02
mu_PAO = (1.0/24.0) * PF['mu_PAO']; eta_NO_PAO = 0.60
K_PHA = 0.01
b_PAO = (0.20/24.0) * PF['b_PAO']; b_PP = (0.20/24.0) * PF['b_PP']
b_PHA = (0.20/24.0) * PF['b_PHA']
K_A_PAO = 4.0 * PF['K_A_PAO']; K_O2_PAO = 0.20; K_NO3_PAO = 0.50; K_NH4_PAO = 0.05
K_PS = 0.20 * PF['K_PS']; K_P_PAO = 0.01; K_ALK_PAO = 0.10

mu_AUT = (1.0/24.0) * PF['mu_AUT']; b_AUT = (0.15/24.0) * PF['b_AUT']
K_O2_AUT = 0.50 * PF['K_O2_AUT']; K_NH4_AUT = 1.0 * PF['K_NH4_AUT']
K_P_AUT = 0.01; K_ALK_AUT = 0.50

k_PRE = (1.0/24.0) * PF['k_PRE']; k_RED = (0.60/24.0) * PF['k_RED']; K_ALK_PRE = 0.50

t_all = np.arange(N_TOTAL, dtype=float)


def diurnal(t, mean, amp_frac, phase_rad, noise_frac=0.12, floor_frac=0.05):
    det = mean * (1.0 + amp_frac * np.sin(2.0 * np.pi / 24.0 * t + phase_rad))
    noise = mean * noise_frac * np.random.randn(len(t))
    return np.maximum(det + noise, mean * floor_frac)


Q_t = diurnal(t_all, Q_avg, amp_frac=0.40, phase_rad=-np.pi / 6.0 + PHASE_SHIFT,
              noise_frac=0.15)

KOI_in_t = diurnal(t_all, 350.0 * IF['KOI_mean'], amp_frac=0.50 * IF['KOI_amp'],
                   phase_rad=np.pi / 4.0 + PHASE_SHIFT, noise_frac=0.14)
NH4_in_t = diurnal(t_all, 35.0 * IF['NH4_mean'], amp_frac=0.45 * IF['NH4_amp'],
                   phase_rad=np.pi / 3.0 + PHASE_SHIFT, noise_frac=0.13)
PO4_in_t = diurnal(t_all, 6.0 * IF['PO4_mean'], amp_frac=0.30 * IF['PO4_amp'],
                   phase_rad=np.pi / 2.8 + PHASE_SHIFT, noise_frac=0.11)

DO_in_t = np.clip(0.5 + 0.15 * np.random.randn(N_TOTAL), 0.0, 1.5)

SI_frac_t = np.clip(
    0.085 + 0.015 * np.sin(2.0 * np.pi / 24.0 * t_all + 1.5)
    + 0.008 * np.random.randn(N_TOTAL),
    0.06, 0.12
)
SI_in_t = np.clip(SI_frac_t * KOI_in_t + 1.5 * np.random.randn(N_TOTAL), 12.0, 80.0)
XI_in_t = np.clip(
    0.10 * KOI_in_t + 8.0 * np.sin(2.0 * np.pi * t_all / (24.0 * 7.0)),
    20.0, 110.0)
bio_COD_in_t = np.maximum(KOI_in_t - SI_in_t - XI_in_t, 60.0)

SA_frac_t = np.clip(0.13 + 0.02 * np.sin(2.0 * np.pi / 24.0 * t_all + 1.2)
                     + 0.010 * np.random.randn(N_TOTAL), 0.08, 0.18)
SF_frac_t = np.clip(0.16 + 0.03 * np.sin(2.0 * np.pi / 24.0 * t_all - 0.2)
                     + 0.015 * np.random.randn(N_TOTAL), 0.10, 0.22)
XS_frac_t = np.clip(1.0 - SA_frac_t - SF_frac_t, 0.18, 0.60)
frac_sum = SA_frac_t + SF_frac_t + XS_frac_t
SA_frac_t /= frac_sum; SF_frac_t /= frac_sum; XS_frac_t /= frac_sum

SA_in_t = bio_COD_in_t * SA_frac_t
SF_in_t = bio_COD_in_t * SF_frac_t
XS_in_t = bio_COD_in_t * XS_frac_t

SNO3_in_t = np.full(N_TOTAL, SNO3_IN_BASE)
SN2_in_t = np.full(N_TOTAL, SN2_IN_BASE)
SALK_in_t = diurnal(t_all, SALK_IN_BASE, amp_frac=0.08, phase_rad=-0.4,
                    noise_frac=0.03, floor_frac=0.60)
XMeOH_in_t = diurnal(t_all, XMeOH_IN_BASE, amp_frac=0.18, phase_rad=0.7,
                     noise_frac=0.05, floor_frac=0.35)

kLa_t = np.full(N_TOTAL, kLa_AE, dtype=float)
kLa_t[:WARMUP_H] = 40.0
for cycle_start in range(WARMUP_H, N_TOTAL, 48):
    shock_h = cycle_start + 24
    for sh in range(shock_h, min(shock_h + 6, N_TOTAL)):
        kLa_t[sh] = kLa_AE * 0.20
ikLa = interp1d(t_all, kLa_t, kind='previous', fill_value='extrapolate')

iQ     = interp1d(t_all, Q_t,        kind='linear', fill_value='extrapolate')
iKOI   = interp1d(t_all, KOI_in_t,   kind='linear', fill_value='extrapolate')
iNH4   = interp1d(t_all, NH4_in_t,   kind='linear', fill_value='extrapolate')
iPO4   = interp1d(t_all, PO4_in_t,   kind='linear', fill_value='extrapolate')
iDO_in = interp1d(t_all, DO_in_t,    kind='linear', fill_value='extrapolate')
iSF    = interp1d(t_all, SF_in_t,    kind='linear', fill_value='extrapolate')
iSA    = interp1d(t_all, SA_in_t,    kind='linear', fill_value='extrapolate')
iXS    = interp1d(t_all, XS_in_t,    kind='linear', fill_value='extrapolate')
iXI    = interp1d(t_all, XI_in_t,    kind='linear', fill_value='extrapolate')
iSNO3  = interp1d(t_all, SNO3_in_t,  kind='linear', fill_value='extrapolate')
iSN2   = interp1d(t_all, SN2_in_t,   kind='linear', fill_value='extrapolate')
iSALK  = interp1d(t_all, SALK_in_t,  kind='linear', fill_value='extrapolate')
iXMeOH = interp1d(t_all, XMeOH_in_t, kind='linear', fill_value='extrapolate')


v_SO2_4  = -(1.0 - YH) / YH
v_SF_4   = -1.0 / YH
v_SNH4_4 = -(- (1.0 / YH) * iN_SF + iN_BM)
v_SPO4_4 = -(- (1.0 / YH) * iP_SF + iP_BM)

v_SO2_5  = -(1.0 - YH) / YH
v_SA_5   = -1.0 / YH
v_SNH4_5 = -iN_BM
v_SPO4_5 = -iP_BM

v_SNO3_denit = -(1.0 - YH) / (iNO3_N2 * YH)
v_SN2_denit  = (1.0 - YH) / (iNO3_N2 * YH)

v_SNH4_lys = -(fXI * iN_XI + (1.0 - fXI) * iN_XS - iN_BM)
v_SPO4_lys = -(fXI * iP_XI + (1.0 - fXI) * iP_XS - iP_BM)
v_XTSS_lys = fXI * iTSS_XI + (1.0 - fXI) * iTSS_XS - iTSS_BM

v_SNO3_PAO_denit = -(1.0 - YPAO) / (YPAO * iNO3_N2)
v_SN2_PAO_denit  = (1.0 - YPAO) / (YPAO * iNO3_N2)
v_XPHA_PAO       = -1.0 / YPAO

v_SO2_18 = -(4.571 - YA) / YA
v_SNH4_18 = -iN_BM - 1.0 / YA
v_SNO3_18 = 1.0 / YA

v_SNO3_PP_anox = -YPHA / iNO3_N2
v_SN2_PP_anox  = YPHA / iNO3_N2

v_XTSS_hyd  = -iTSS_XS
v_XTSS_grow = iTSS_BM
v_XTSS_10   = -YPO4 * iTSS_XPP + iTSS_XPHA
v_XTSS_11   = iTSS_XPP - YPHA * iTSS_XPHA
v_XTSS_13   = iTSS_BM - (1.0 / YPAO) * iTSS_XPHA
v_XTSS_16   = -iTSS_XPP
v_XTSS_17   = -iTSS_XPHA
v_XTSS_20   = fMeOH + fMeP


def compute_zone_rates(y_zone):
    vals = np.maximum(y_zone, 0.0)
    (SO2, SF, SA, SNH4, SNO3, SPO4, SALK, SN2,
     XI, XS, XH, XPAO, XPP, XPHA, XAUT, XTSS, XMeOH, XMeP) = vals

    XPAO_s = max(XPAO, EPS)
    XH_s   = max(XH, EPS)

    def mon(s, k):
        return max(s, 0.0) / (k + max(s, 0.0) + EPS)

    def inh(s, k):
        return k / (k + max(s, 0.0) + EPS)

    sf_sa = SF + SA + EPS
    sw_SF = SF / sf_sa
    sw_SA = SA / sf_sa

    XPP_ratio  = XPP / XPAO_s
    XPHA_ratio = XPHA / XPAO_s
    XS_ratio   = XS / XH_s

    m_XPP_ratio  = XPP_ratio / (K_PP + XPP_ratio + EPS)
    m_XPHA_ratio = XPHA_ratio / (K_PHA + XPHA_ratio + EPS)
    hydro_access = XS_ratio / (K_X + XS_ratio + EPS)

    pp_inhib_num = max(K_MAX - XPP_ratio, 0.0)
    pp_inhib = pp_inhib_num / (K_iPP + pp_inhib_num + EPS)

    rho1  = k_hyd * mon(SO2, K_O2_hyd) * hydro_access * XH
    rho2  = k_hyd * eta_NO_hyd * inh(SO2, K_O2_hyd) * mon(SNO3, K_NO3_hyd) \
            * hydro_access * XH
    rho3  = k_hyd * eta_fe_hyd * inh(SO2, K_O2_hyd) * inh(SNO3, K_NO3_hyd) \
            * hydro_access * XH
    rho4  = mu_H * mon(SO2, K_O2_H) * mon(SF, K_F) * sw_SF \
            * mon(SNH4, K_NH4_H) * mon(SPO4, K_P_H) * mon(SALK, K_ALK_H) * XH
    rho5  = mu_H * mon(SO2, K_O2_H) * mon(SA, K_A_H) * sw_SA \
            * mon(SNH4, K_NH4_H) * mon(SPO4, K_P_H) * mon(SALK, K_ALK_H) * XH
    rho6  = mu_H * eta_NO_H * inh(SO2, K_O2_H) * mon(SNO3, K_NO3_H) \
            * mon(SF, K_F) * sw_SF \
            * mon(SNH4, K_NH4_H) * mon(SPO4, K_P_H) * mon(SALK, K_ALK_H) * XH
    rho7  = mu_H * eta_NO_H * inh(SO2, K_O2_H) * mon(SNO3, K_NO3_H) \
            * mon(SA, K_A_H) * sw_SA \
            * mon(SNH4, K_NH4_H) * mon(SPO4, K_P_H) * mon(SALK, K_ALK_H) * XH
    rho8  = q_fe * inh(SO2, K_O2_H) * inh(SNO3, K_NO3_H) \
            * mon(SF, K_fe) * mon(SALK, K_ALK_H) * XH
    rho9  = b_H * XH
    rho10 = q_PHA * mon(SA, K_A_PAO) * mon(SALK, K_ALK_PAO) \
            * m_XPP_ratio * XPAO
    rho11 = q_PP * mon(SO2, K_O2_PAO) * mon(SPO4, K_PS) \
            * mon(SALK, K_ALK_PAO) * m_XPHA_ratio * pp_inhib * XPAO
    rho12 = q_PP * eta_NO_PAO * mon(SNO3, K_NO3_PAO) * inh(SO2, K_O2_PAO) \
            * mon(SPO4, K_PS) * mon(SALK, K_ALK_PAO) \
            * m_XPHA_ratio * pp_inhib * XPAO
    rho13 = mu_PAO * mon(SO2, K_O2_PAO) * mon(SNH4, K_NH4_PAO) \
            * mon(SPO4, K_P_PAO) * mon(SALK, K_ALK_PAO) \
            * m_XPHA_ratio * XPAO
    rho14 = mu_PAO * eta_NO_PAO * inh(SO2, K_O2_PAO) * mon(SNO3, K_NO3_PAO) \
            * mon(SNH4, K_NH4_PAO) * mon(SPO4, K_P_PAO) * mon(SALK, K_ALK_PAO) \
            * m_XPHA_ratio * XPAO
    rho15 = b_PAO * XPAO * mon(SALK, K_ALK_PAO)
    rho16 = b_PP * mon(SALK, K_ALK_PAO) * XPP_ratio * XPAO
    rho17 = b_PHA * mon(SALK, K_ALK_PAO) * XPHA_ratio * XPAO
    rho18 = mu_AUT * mon(SO2, K_O2_AUT) * mon(SNH4, K_NH4_AUT) \
            * mon(SPO4, K_P_AUT) * mon(SALK, K_ALK_AUT) * XAUT
    rho19 = b_AUT * XAUT
    rho20 = k_PRE * SPO4 * XMeOH
    rho21 = k_RED * XMeP * mon(SALK, K_ALK_PRE)

    r_hyd = rho1 + rho2 + rho3

    drates = np.zeros(18)

    drates[0] = (v_SO2_4 * (rho4 + rho5)
                 + (-YPHA) * rho11
                 + v_SO2_4 * rho13
                 + v_SO2_18 * rho18)

    drates[1] = ((1.0 - fSI) * r_hyd
                 + v_SF_4 * rho4
                 + v_SF_4 * rho6
                 - 1.0 * rho8)

    drates[2] = (v_SA_5 * rho5
                 + v_SA_5 * rho7
                 + 1.0 * rho8
                 - 1.0 * rho10
                 + 1.0 * rho17)

    v_SNH4_hyd = -((1.0 - fSI) * iN_SF + fSI * iN_SI - iN_XS)
    drates[3] = (v_SNH4_hyd * r_hyd
                 + v_SNH4_4 * (rho4 + rho6)
                 + v_SNH4_5 * (rho5 + rho7)
                 + iN_SF * rho8
                 + v_SNH4_lys * (rho9 + rho15 + rho19)
                 + (-iN_BM) * (rho13 + rho14)
                 + v_SNH4_18 * rho18)

    drates[4] = (v_SNO3_denit * (rho6 + rho7)
                 + v_SNO3_PP_anox * rho12
                 + v_SNO3_PAO_denit * rho14
                 + v_SNO3_18 * rho18)

    v_SPO4_hyd = -((1.0 - fSI) * iP_SF + fSI * iP_SI - iP_XS)
    drates[5] = (v_SPO4_hyd * r_hyd
                 + v_SPO4_4 * (rho4 + rho6)
                 + (-iP_BM) * (rho5 + rho7)
                 + iP_SF * rho8
                 + v_SPO4_lys * (rho9 + rho15 + rho19)
                 + YPO4 * rho10
                 - 1.0 * (rho11 + rho12)
                 + (-iP_BM) * (rho13 + rho14)
                 + 1.0 * rho16
                 + (-iP_BM) * rho18
                 - 1.0 * rho20
                 + 1.0 * rho21)

    iC_NHx = 0.071; iC_NOx = -0.071; iC_PO4 = -0.048; iC_VFA = -0.016; iC_PP = -0.032

    drates[6] = (iC_NHx * v_SNH4_hyd * r_hyd + iC_PO4 * v_SPO4_hyd * r_hyd
                 + (iC_NHx * v_SNH4_4 + iC_PO4 * v_SPO4_4) * (rho4 + rho6)
                 + (iC_NHx * v_SNH4_5 + iC_PO4 * (-iP_BM)) * (rho5 + rho7)
                 + (iC_NHx * iN_SF + iC_PO4 * iP_SF + iC_VFA) * rho8
                 + iC_NOx * v_SNO3_denit * (rho6 + rho7)
                 + (iC_NHx * v_SNH4_lys + iC_PO4 * v_SPO4_lys) * (rho9 + rho15 + rho19)
                 + (-iC_VFA + iC_PO4 * YPO4 - iC_PP * YPO4) * rho10
                 + (-iC_PO4 + iC_PP) * (rho11 + rho12)
                 + iC_NOx * v_SNO3_PP_anox * rho12
                 + (iC_NHx * (-iN_BM) + iC_PO4 * (-iP_BM)) * (rho13 + rho14)
                 + iC_NOx * v_SNO3_PAO_denit * rho14
                 + (iC_PO4 - iC_PP) * rho16
                 + iC_VFA * rho17
                 + (iC_NHx * v_SNH4_18 + iC_NOx * v_SNO3_18 + iC_PO4 * (-iP_BM)) * rho18
                 + (-iC_PO4) * rho20
                 + iC_PO4 * rho21)

    drates[7] = (v_SN2_denit * (rho6 + rho7)
                 + v_SN2_PP_anox * rho12
                 + v_SN2_PAO_denit * rho14)

    drates[8] = fXI * (rho9 + rho15 + rho19)

    drates[9] = (-1.0 * r_hyd
                 + (1.0 - fXI) * (rho9 + rho15 + rho19))

    drates[10] = (1.0 * (rho4 + rho5 + rho6 + rho7) - 1.0 * rho9)

    drates[11] = (1.0 * (rho13 + rho14) - 1.0 * rho15)

    drates[12] = (-YPO4 * rho10 + 1.0 * (rho11 + rho12) - 1.0 * rho16)

    drates[13] = (1.0 * rho10
                  - YPHA * (rho11 + rho12)
                  + v_XPHA_PAO * (rho13 + rho14)
                  - 1.0 * rho17)

    drates[14] = (1.0 * rho18 - 1.0 * rho19)

    drates[15] = (v_XTSS_hyd * r_hyd
                  + v_XTSS_grow * (rho4 + rho5 + rho6 + rho7)
                  + v_XTSS_lys * (rho9 + rho15 + rho19)
                  + v_XTSS_10 * rho10
                  + v_XTSS_11 * (rho11 + rho12)
                  + v_XTSS_13 * (rho13 + rho14)
                  + v_XTSS_16 * rho16
                  + v_XTSS_17 * rho17
                  + v_XTSS_grow * rho18
                  + v_XTSS_20 * rho20
                  + (-v_XTSS_20) * rho21)

    drates[16] = fMeOH * rho20 + (-fMeOH) * rho21

    drates[17] = fMeP * rho20 + (-fMeP) * rho21

    drates = np.where(np.isfinite(drates), drates, 0.0)
    return drates


def asm2d_a2o_ode(t_now, y_combined):
    y_an = y_combined[0:N_STATES]
    y_ax = y_combined[N_STATES:2*N_STATES]
    y_ae = y_combined[2*N_STATES:3*N_STATES]

    y_an = np.maximum(y_an, 0.0)
    y_ax = np.maximum(y_ax, 0.0)
    y_ae = np.maximum(y_ae, 0.0)

    Q     = max(float(iQ(t_now)), 50.0)
    Q_IR  = Q_IR_factor * Q
    Q_RAS = Q_RAS_factor * Q

    inputs = np.zeros(18)
    inputs[0]  = max(float(iDO_in(t_now)), 0.0)
    inputs[1]  = max(float(iSF(t_now)), 0.0)
    inputs[2]  = max(float(iSA(t_now)), 0.0)
    inputs[3]  = max(float(iNH4(t_now)), 0.0)
    inputs[4]  = max(float(iSNO3(t_now)), 0.0)
    inputs[5]  = max(float(iPO4(t_now)), 0.0)
    inputs[6]  = max(float(iSALK(t_now)), 0.0)
    inputs[7]  = max(float(iSN2(t_now)), 0.0)
    inputs[8]  = max(float(iXI(t_now)), 0.0)
    inputs[9]  = max(float(iXS(t_now)), 0.0)
    inputs[16] = max(float(iXMeOH(t_now)), 0.0)
    inputs[15] = iTSS_XI * inputs[8] + iTSS_XS * inputs[9] + inputs[16]

    C_RAS = np.zeros(18)
    for i in SOLUBLE_IDX:
        C_RAS[i] = y_ae[i]
    C_RAS[0] = 0.0
    C_RAS[4] = 0.0
    for i in PARTICULATE_IDX:
        thickening = (solids_capture * Q + Q_RAS) / max(Q_RAS, 1.0)
        C_RAS[i] = y_ae[i] * thickening

    rates_an = compute_zone_rates(y_an)
    rates_ax = compute_zone_rates(y_ax)
    rates_ae = compute_zone_rates(y_ae)

    Q_out_an = Q + Q_RAS
    d_an = np.zeros(18)
    for i in range(18):
        flux_in  = Q * inputs[i] + Q_RAS * C_RAS[i]
        flux_out = Q_out_an * y_an[i]
        d_an[i]  = (flux_in - flux_out) / V_AN + rates_an[i]
    d_an[0] -= 5.0 * y_an[0]

    Q_out_ax = Q + Q_RAS + Q_IR
    y_ae_ir = y_ae.copy()
    y_ae_ir[0] = 0.0
    d_ax = np.zeros(18)
    for i in range(18):
        flux_in  = Q_out_an * y_an[i] + Q_IR * y_ae_ir[i]
        flux_out = Q_out_ax * y_ax[i]
        d_ax[i]  = (flux_in - flux_out) / V_AX + rates_ax[i]
    d_ax[0] -= 2.0 * y_ax[0]

    d_ae = np.zeros(18)
    for i in range(18):
        flux_in  = Q_out_ax * y_ax[i]
        flux_out = Q_out_ax * y_ae[i]
        d_ae[i]  = (flux_in - flux_out) / V_AE + rates_ae[i]
    kLa_now = max(float(ikLa(t_now)), 0.1)
    d_ae[0] += kLa_now * (DO_sat - y_ae[0])

    SRT_AE_rate = V_TOTAL / (V_AE * SRT_h)
    for i in PARTICULATE_IDX:
        d_ae[i] -= y_ae[i] * SRT_AE_rate

    result = np.concatenate([d_an, d_ax, d_ae])
    result = np.where(np.isfinite(result), result, 0.0)
    return result


y0_an = np.array([
    0.0, 8.0, 26.0, 25.0, 0.1, 15.0, 7.0, 20.0,
    80.0, 100.0, 1000.0, 200.0, 60.0, 120.0, 300.0, 3200.0, 8.0, 3.0,
], dtype=float)
y0_ax = np.array([
    0.0, 2.0, 3.5, 12.0, 13.5, 3.0, 6.5, 18.0,
    80.0, 80.0, 1000.0, 200.0, 60.0, 100.0, 300.0, 3200.0, 9.0, 2.5,
], dtype=float)
y0_ae = np.array([
    2.5, 0.5, 0.7, 3.0, 25.0, 0.8, 7.0, 20.0,
    80.0, 40.0, 1000.0, 250.0, 90.0, 20.0, 300.0, 3200.0, 9.0, 4.0,
], dtype=float)
y0_combined = np.concatenate([y0_an, y0_ax, y0_ae])

print(f"\n[1/4] OOD Multi-zone A2O ODE simulasyonu baslatiliyor (BDF, stiff)...")

ivp_result = solve_ivp(
    asm2d_a2o_ode,
    t_span=(0, N_TOTAL - 1),
    y0=y0_combined,
    method='BDF',
    t_eval=t_all,
    rtol=1e-4,
    atol=1e-6,
    max_step=0.5,
)
if not ivp_result.success:
    print(f"      UYARI: {ivp_result.message}")
sol = np.maximum(ivp_result.y.T, 0.0)
print("      ODE cozuldu.")

sol_an = sol[:, 0:N_STATES]
sol_ax = sol[:, N_STATES:2*N_STATES]
sol_ae = sol[:, 2*N_STATES:3*N_STATES]

sol_an = sol_an[WARMUP_H:]
sol_ax = sol_ax[WARMUP_H:]
sol_ae = sol_ae[WARMUP_H:]
t_out = (t_all[WARMUP_H:] - WARMUP_H).astype(int)

ae_df = pd.DataFrame(sol_ae, columns=STATE_NAMES)

KOI_particulate_eff = (1.0 - solids_capture) * (
    ae_df['XI'].values + ae_df['XS'].values + ae_df['XH'].values
    + ae_df['XPAO'].values + ae_df['XPHA'].values + ae_df['XAUT'].values
)
SI_eff_t = SI_in_t[WARMUP_H:]
KOI_cikis = ae_df['SF'].values + ae_df['SA'].values + SI_eff_t + KOI_particulate_eff
NH4_cikis = ae_df['SNH4'].values
PO4_cikis = ae_df['SPO4'].values

KOI_in_out = KOI_in_t[WARMUP_H:]
NH4_in_out = NH4_in_t[WARMUP_H:]
PO4_in_out = PO4_in_t[WARMUP_H:]
DO_out = ae_df['SO2'].values.copy()

df = pd.DataFrame({
    't_saat': t_out,
    'KOI_giris': np.round(KOI_in_out, 3),
    'NH4_giris': np.round(NH4_in_out, 3),
    'PO4_giris': np.round(PO4_in_out, 3),
    'DO': np.round(DO_out, 3),
    'KOI_cikis': np.round(KOI_cikis, 3),
    'NH4_cikis': np.round(NH4_cikis, 3),
    'PO4_cikis': np.round(PO4_cikis, 3),
})
for col in STATE_NAMES:
    df[col] = np.round(ae_df[col].values, 3)

csv_path = 'synthetic_asm2d_physics_ood.csv'
df.to_csv(csv_path, index=False)
print(f"\n[2/4] '{csv_path}' kaydedildi --> {len(df):,} satir")

multizone_path = 'synthetic_asm2d_multizone_ood.csv'
mz_df = pd.DataFrame({'t_saat': t_out})
for zone, sol_z in [('AN', sol_an), ('AX', sol_ax), ('AE', sol_ae)]:
    for i, name in enumerate(STATE_NAMES):
        mz_df[f'{zone}_{name}'] = np.round(sol_z[:, i], 3)
mz_df.to_csv(multizone_path, index=False)
print(f"      '{multizone_path}' kaydedildi")

pert_log = {
    'OOD_SEED': OOD_SEED, 'PERT_FRAC': PERT_FRAC, 'INFL_FRAC': INFL_FRAC,
    'PHASE_SHIFT_rad': PHASE_SHIFT,
    'kinetic_factors': PF, 'influent_factors': IF,
}
with open('ood_perturbation_factors.json', 'w', encoding='utf-8') as f:
    json.dump(pert_log, f, indent=2)
print("      'ood_perturbation_factors.json' kaydedildi")

print("\n[3/4] OOD Veri Dogrulama Istatistikleri:")
print("-" * 72)
summary_cols = ['KOI_giris', 'NH4_giris', 'PO4_giris', 'DO',
                'KOI_cikis', 'NH4_cikis', 'PO4_cikis']
print(df[summary_cols].describe().round(2).to_string())
print("-" * 72)

giderim_KOI = (1.0 - df['KOI_cikis'].mean() / df['KOI_giris'].mean()) * 100
giderim_NH4 = (1.0 - df['NH4_cikis'].mean() / df['NH4_giris'].mean()) * 100
giderim_PO4 = (1.0 - df['PO4_cikis'].mean() / df['PO4_giris'].mean()) * 100

print("\nFiziksel Iliski Kontrolleri:")
print(f"  KOI giderim verimi    : {giderim_KOI:.1f}%")
print(f"  NH4 giderim verimi    : {giderim_NH4:.1f}%")
print(f"  PO4 giderim verimi    : {giderim_PO4:.1f}%")

ab_koi = (df['KOI_cikis'] < 125).mean() * 100
ab_nh4 = (df['NH4_cikis'] < 10).mean() * 100
ab_po4 = (df['PO4_cikis'] < 2).mean() * 100
print(f"\n  AB Uyum Oranlari (OOD Verisi):")
print(f"    KOI < 125 mg/L : {ab_koi:.1f}%")
print(f"    NH4 < 10  mg/L : {ab_nh4:.1f}%")
print(f"    PO4 < 2   mg/L : {ab_po4:.1f}%")

xpao_ae_mean = sol_ae[:, S_IDX['XPAO']].mean()
spo4_an_mean = sol_an[:, S_IDX['SPO4']].mean()
spo4_ae_mean = sol_ae[:, S_IDX['SPO4']].mean()
print(f"\n  EBPR Dogrulama:")
print(f"    XPAO (aerobik) ort = {xpao_ae_mean:.1f} mg/L  {'OK' if xpao_ae_mean > 50 else 'YETERSIZ'}")
print(f"    SPO4 (anaerobik)   = {spo4_an_mean:.2f} mg/L  {'OK' if spo4_an_mean > spo4_ae_mean else 'YETERSIZ'}")
print(f"    KOI std (cikis)    = {df['KOI_cikis'].std():.2f} mg/L  {'OK' if df['KOI_cikis'].std() > 5 else 'DAR'}")

print("\n[4/4] Grafik olusturuluyor (OOD)...")
fig, axes = plt.subplots(8, 1, figsize=(14, 22), sharex=False)
plot_cols = ['KOI_giris', 'NH4_giris', 'PO4_giris', 'DO',
             'KOI_cikis', 'NH4_cikis', 'PO4_cikis', 'SNO3']
plot_labels = ['COD Influent [mg/L]', 'NH4 Influent [mg/L]', 'PO4 Influent [mg/L]',
               'DO Aerobic [mg/L]', 'COD Effluent [mg/L]', 'NH4 Effluent [mg/L]',
               'PO4 Effluent [mg/L]', 'SNO3 Aerobic [mg N/L]']
plot_colors = ['#2563EB', '#16A34A', '#0EA5E9', '#DC2626',
               '#D97706', '#7C3AED', '#059669', '#475569']
show_h = 336
for ax, col, lbl, clr in zip(axes, plot_cols, plot_labels, plot_colors):
    ax.plot(t_out[:show_h], df[col].values[:show_h], color=clr, lw=1.0, alpha=0.9)
    ax.set_ylabel(lbl, fontsize=9)
    ax.grid(True, alpha=0.28)
    ax.set_xlim(0, show_h)
axes[-1].set_xlabel("Time [h]", fontsize=10)
fig.suptitle("ASM2d A2O OOD Scenario -- First 14 Days", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('asm2d_ode_timeseries_ood.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 72)
print("  OOD URETIMI TAMAMLANDI")
print(f"  Ana CSV : {csv_path} ({len(df):,} satir)")
print("=" * 72)
