import json
import numpy as np
import torch
from tqdm import tqdm

# ── toggles ──────────────────────────────────────────────────────────────
USE_SQRT_PT = True          # clip+sqrt+standardize vs clip+standardize
MTD_METHOD = "B"            # "A" = indicator + neutral fill | "B" = raw fill, no indicator

# ── fixed hyperparameters (decisions, not statistics) ────────────────────
PT_CLIP_PERCENTILE = 99.6
TIME_SENTINEL, TIME_ERR_SENTINEL = 0.0, -1.0
TIME_FILL_B, TIME_ERR_FILL_B = 9.0, -0.001
TIME_FILL_A, TIME_ERR_FILL_A = 0.0, 0.0

# ── track feature column indices (feature_trk order) ────────────────────
IDX_ETA, IDX_PHI, IDX_X, IDX_Y, IDX_Z = 0, 1, 2, 3, 4
IDX_P, IDX_PT, IDX_NHITS = 5, 6, 7
IDX_TIME, IDX_TIME_ERR = 8, 9

# ── trkst fixed hyperparameters (decisions, not statistics) ──────────────
TS_TIME_SENTINEL, TS_TIME_ERR_SENTINEL = -99.0, -1.0
TS_TIME_CLAMP_MIN, TS_TIME_CLAMP_MAX = 9.0, 18.0
TS_TIME_FILL = 9.0
TS_TIME_ERR_FILL_MARGIN_SIGMA = 4.0

# ── trkst feature column indices (feature_trkst order, offset by feature_trk) ──
TS_OFFSET = 10  # len(feature_trk)
IDX_TS_ETA, IDX_TS_PHI, IDX_TS_X, IDX_TS_Y, IDX_TS_Z = (TS_OFFSET + i for i in range(5))
IDX_TS_ENERGY = TS_OFFSET + 5
IDX_TS_TIME = TS_OFFSET + 6
IDX_TS_TIME_ERR = TS_OFFSET + 7
IDX_TS_EM_ENERGY = TS_OFFSET + 8
IDX_TS_EM_PT = TS_OFFSET + 9
IDX_TS_PT = TS_OFFSET + 10

# ── trk-ts edge feature column indices (edge_attr order) ─────────────────
EDGE_IDX_DETA, EDGE_IDX_DPHI, EDGE_IDX_DR = 0, 1, 2
EDGE_IDX_DE, EDGE_IDX_ERATIO = 3, 4
EDGE_IDX_DT, EDGE_IDX_DTSIG = 5, 6
EDGE_IDX_TYPE = 7
EDGE_IDX_TIME_INDICATOR = 8  # new column, appended

EDGE_TRK_P_IDX = 5  # track_p column within feature_trk (raw x, source node)

# ── trk-ts edge fixed hyperparameters (decisions, not statistics) ────────
EDGE_DT_SENTINEL = 0.0
EDGE_DT_FILL_MARGIN_SIGMA = 4.0
EDGE_DT_FILL_SIDE = "below"  # "below" real_min, or "above" real_max
EDGE_DT_Z_CLIP_MIN, EDGE_DT_Z_CLIP_MAX = -5.0, 7.0

# ── ts-ts edge fixed hyperparameters (decisions, not statistics) ─────────
TT_DT_CLIP_VALUE = 2.0
TT_TARGET_Z = -6.0


def fit_track_stats(dataset):
    cols = {"eta": [], "phi": [], "x": [], "y": [], "nhits": [], "pt": [], "time": [], "time_err": []}

    for gr in tqdm(dataset, desc="fitting track stats"):
        n_trk = gr.n_trk
        if n_trk == 0:
            continue
        trk = gr.x[:n_trk]
        cols["eta"].extend(trk[:, IDX_ETA].tolist())
        cols["phi"].extend(trk[:, IDX_PHI].tolist())
        cols["x"].extend(trk[:, IDX_X].tolist())
        cols["y"].extend(trk[:, IDX_Y].tolist())
        cols["nhits"].extend(trk[:, IDX_NHITS].tolist())
        cols["pt"].extend(trk[:, IDX_PT].tolist())
        cols["time"].extend(trk[:, IDX_TIME].tolist())
        cols["time_err"].extend(trk[:, IDX_TIME_ERR].tolist())

    eta = np.array(cols["eta"])
    phi = np.array(cols["phi"])
    x_ = np.array(cols["x"])
    y_ = np.array(cols["y"])
    nhits = np.array(cols["nhits"])
    pt = np.array(cols["pt"])
    time = np.array(cols["time"])
    err = np.array(cols["time_err"])

    time_mask = time != TIME_SENTINEL
    real_time = time[time_mask]
    real_err = err[time_mask]

    pt_clip_value = float(np.percentile(pt, PT_CLIP_PERCENTILE))
    pt_clipped = np.clip(pt, a_min=None, a_max=pt_clip_value)
    pt_transformed = np.sqrt(pt_clipped) if USE_SQRT_PT else pt_clipped

    stats = {
        "eta_mean": eta.mean(), "eta_std": eta.std(),
        "phi_mean": phi.mean(), "phi_std": phi.std(),
        "x_mean": x_.mean(), "x_std": x_.std(),
        "y_mean": y_.mean(), "y_std": y_.std(),
        "nhits_mean": nhits.mean(), "nhits_std": nhits.std(),
        "pt_clip_value": pt_clip_value,
        "pt_mean": pt_transformed.mean(), "pt_std": pt_transformed.std(),
        "time_mean": real_time.mean(), "time_std": real_time.std(),
        "time_err_mean": real_err.mean(), "time_err_std": real_err.std(),
    }
    return stats


def fit_trkst_stats(dataset):
    cols = {"eta": [], "phi": [], "x": [], "y": [], "z": [],
            "energy": [], "em_energy": [], "time": [], "time_err": []}

    for gr in tqdm(dataset, desc="fitting trkst stats"):
        n_trk = gr.n_trk
        ts = gr.x[n_trk:]
        if ts.shape[0] == 0:
            continue
        cols["eta"].extend(ts[:, IDX_TS_ETA].tolist())
        cols["phi"].extend(ts[:, IDX_TS_PHI].tolist())
        cols["x"].extend(ts[:, IDX_TS_X].tolist())
        cols["y"].extend(ts[:, IDX_TS_Y].tolist())
        cols["z"].extend(ts[:, IDX_TS_Z].tolist())
        cols["energy"].extend(ts[:, IDX_TS_ENERGY].tolist())
        cols["em_energy"].extend(ts[:, IDX_TS_EM_ENERGY].tolist())
        cols["time"].extend(ts[:, IDX_TS_TIME].tolist())
        cols["time_err"].extend(ts[:, IDX_TS_TIME_ERR].tolist())

    eta = np.array(cols["eta"])
    phi = np.array(cols["phi"])
    x_ = np.array(cols["x"])
    y_ = np.array(cols["y"])
    z_ = np.array(cols["z"])
    energy = np.array(cols["energy"])
    em_energy = np.array(cols["em_energy"])
    time = np.array(cols["time"])
    err = np.array(cols["time_err"])

    log_energy = np.log1p(energy)
    log_em_energy = np.log1p(em_energy)

    time_mask = time != TS_TIME_SENTINEL
    real_time = np.clip(time[time_mask], TS_TIME_CLAMP_MIN, TS_TIME_CLAMP_MAX)
    real_err = np.sqrt(err[time_mask])

    err_fill = float(real_err.min() - TS_TIME_ERR_FILL_MARGIN_SIGMA * real_err.std())

    stats = {
        "ts_eta_mean": eta.mean(), "ts_eta_std": eta.std(),
        "ts_phi_mean": phi.mean(), "ts_phi_std": phi.std(),
        "ts_x_mean": x_.mean(), "ts_x_std": x_.std(),
        "ts_y_mean": y_.mean(), "ts_y_std": y_.std(),
        "ts_z_mean": z_.mean(), "ts_z_std": z_.std(),
        "ts_energy_mean": log_energy.mean(), "ts_energy_std": log_energy.std(),
        "ts_em_energy_mean": log_em_energy.mean(), "ts_em_energy_std": log_em_energy.std(),
        "ts_time_mean": real_time.mean(), "ts_time_std": real_time.std(),
        "ts_time_err_mean": real_err.mean(), "ts_time_err_std": real_err.std(),
        "ts_time_err_fill": err_fill,
    }
    return stats


def fit_trk_ts_edge_stats(dataset):
    cols = {"deta": [], "dphi": [], "dR": [], "E_ratio": [], "deltaTime": [], "dtime_sig": []}

    for gr in tqdm(dataset, desc="fitting trk-ts edge stats"):
        mask = gr.edge_attr[:, EDGE_IDX_TYPE] == 0
        if mask.sum() == 0:
            continue
        e = gr.edge_attr[mask]
        cols["deta"].extend(e[:, EDGE_IDX_DETA].tolist())
        cols["dphi"].extend(e[:, EDGE_IDX_DPHI].tolist())
        cols["dR"].extend(e[:, EDGE_IDX_DR].tolist())
        cols["E_ratio"].extend(e[:, EDGE_IDX_ERATIO].tolist())
        cols["deltaTime"].extend(e[:, EDGE_IDX_DT].tolist())
        cols["dtime_sig"].extend(e[:, EDGE_IDX_DTSIG].tolist())

    deta = np.array(cols["deta"])
    dphi = np.array(cols["dphi"])
    dR = np.array(cols["dR"])
    E_ratio = np.array(cols["E_ratio"])
    dt = np.array(cols["deltaTime"])
    dtsig = np.array(cols["dtime_sig"])

    log_E_ratio = np.log10(E_ratio)

    dt_mask = dt != EDGE_DT_SENTINEL
    real_dt = dt[dt_mask]
    real_dtsig_log = np.log10(dtsig[dt_mask])

    if EDGE_DT_FILL_SIDE == "below":
        dt_fill = float(real_dt.min() - EDGE_DT_FILL_MARGIN_SIGMA * real_dt.std())
    else:
        dt_fill = float(real_dt.max() + EDGE_DT_FILL_MARGIN_SIGMA * real_dt.std())
    dtsig_fill_log = float(real_dtsig_log.min() - EDGE_DT_FILL_MARGIN_SIGMA * real_dtsig_log.std())

    stats = {
        "et_deta_mean": deta.mean(), "et_deta_std": deta.std(),
        "et_dphi_mean": dphi.mean(), "et_dphi_std": dphi.std(),
        "et_dR_mean": dR.mean(), "et_dR_std": dR.std(),
        "et_E_ratio_mean": log_E_ratio.mean(), "et_E_ratio_std": log_E_ratio.std(),
        "et_dt_mean": real_dt.mean(), "et_dt_std": real_dt.std(), "et_dt_fill": dt_fill,
        "et_dtsig_mean": real_dtsig_log.mean(), "et_dtsig_std": real_dtsig_log.std(),
        "et_dtsig_fill": dtsig_fill_log,
    }
    return stats


def fit_ts_ts_edge_stats(dataset):
    cols = {"deta": [], "dphi": [], "dR": [], "dE": [], "E_ratio": [],
            "deltaTime": [], "dtime_sig": []}

    for gr in tqdm(dataset, desc="fitting ts-ts edge stats"):
        mask = gr.edge_attr[:, EDGE_IDX_TYPE] == 1
        if mask.sum() == 0:
            continue
        e = gr.edge_attr[mask]
        cols["deta"].extend(e[:, EDGE_IDX_DETA].tolist())
        cols["dphi"].extend(e[:, EDGE_IDX_DPHI].tolist())
        cols["dR"].extend(e[:, EDGE_IDX_DR].tolist())
        cols["dE"].extend(e[:, EDGE_IDX_DE].tolist())
        cols["E_ratio"].extend(e[:, EDGE_IDX_ERATIO].tolist())
        cols["deltaTime"].extend(e[:, EDGE_IDX_DT].tolist())
        cols["dtime_sig"].extend(e[:, EDGE_IDX_DTSIG].tolist())

    deta = np.array(cols["deta"])
    dphi = np.array(cols["dphi"])
    dR = np.array(cols["dR"])
    dE = np.array(cols["dE"])
    E_ratio = np.array(cols["E_ratio"])
    dt = np.array(cols["deltaTime"])
    dtsig = np.array(cols["dtime_sig"])

    log_dE = np.sign(dE) * np.log1p(np.abs(dE))
    log_E_ratio = np.log10(E_ratio)

    dt_mask = dt != EDGE_DT_SENTINEL
    real_dt = dt[dt_mask]
    real_dt_clipped = np.clip(real_dt, -TT_DT_CLIP_VALUE, TT_DT_CLIP_VALUE)
    real_dt_mean, real_dt_std = real_dt_clipped.mean(), real_dt_clipped.std()
    dt_fill = float(real_dt_mean + TT_TARGET_Z * real_dt_std)

    real_dtsig = dtsig[dt_mask]
    log_dtsig = np.sign(real_dtsig) * np.log1p(np.abs(real_dtsig))
    dtsig_mean, dtsig_std = log_dtsig.mean(), log_dtsig.std()
    dtsig_fill = float(dtsig_mean + TT_TARGET_Z * dtsig_std)

    stats = {
        "ett_deta_mean": deta.mean(), "ett_deta_std": deta.std(),
        "ett_dphi_mean": dphi.mean(), "ett_dphi_std": dphi.std(),
        "ett_dR_mean": dR.mean(), "ett_dR_std": dR.std(),
        "ett_dE_mean": log_dE.mean(), "ett_dE_std": log_dE.std(),
        "ett_E_ratio_mean": log_E_ratio.mean(), "ett_E_ratio_std": log_E_ratio.std(),
        "ett_dt_mean": real_dt_mean, "ett_dt_std": real_dt_std, "ett_dt_fill": dt_fill,
        "ett_dtsig_mean": dtsig_mean, "ett_dtsig_std": dtsig_std, "ett_dtsig_fill": dtsig_fill,
    }
    return stats


def save_stats(stats, path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def preprocess_track_features(data, stats):
    n_trk = data.n_trk
    if n_trk == 0:
        return data

    x = data.x
    trk = x[:n_trk]

    trk[:, IDX_ETA] = (trk[:, IDX_ETA] - stats["eta_mean"]) / stats["eta_std"]
    trk[:, IDX_PHI] = (trk[:, IDX_PHI] - stats["phi_mean"]) / stats["phi_std"]
    trk[:, IDX_X] = (trk[:, IDX_X] - stats["x_mean"]) / stats["x_std"]
    trk[:, IDX_Y] = (trk[:, IDX_Y] - stats["y_mean"]) / stats["y_std"]
    trk[:, IDX_NHITS] = (trk[:, IDX_NHITS] - stats["nhits_mean"]) / stats["nhits_std"]

    # dropped features -- zeroed out, kept in place for now
    trk[:, IDX_Z] = 0.0
    trk[:, IDX_P] = 0.0

    # track_hgcal_pt
    pt = torch.clamp(trk[:, IDX_PT], max=stats["pt_clip_value"])
    if USE_SQRT_PT:
        pt = torch.sqrt(pt)
    pt = (pt - stats["pt_mean"]) / stats["pt_std"]
    trk[:, IDX_PT] = pt

    # track_time_mtd / track_time_mtd_err
    time = trk[:, IDX_TIME]
    err = trk[:, IDX_TIME_ERR]
    mask = time != TIME_SENTINEL

    if MTD_METHOD == "A":
        time_out = torch.zeros_like(time)
        err_out = torch.zeros_like(err)
        time_out[mask] = (time[mask] - stats["time_mean"]) / stats["time_std"]
        err_out[mask] = (err[mask] - stats["time_err_mean"]) / stats["time_err_std"]
        time_out[~mask] = TIME_FILL_A
        err_out[~mask] = TIME_ERR_FILL_A
    else:
        time_filled = torch.where(mask, time, torch.full_like(time, TIME_FILL_B))
        err_filled = torch.where(mask, err, torch.full_like(err, TIME_ERR_FILL_B))
        time_out = (time_filled - stats["time_mean"]) / stats["time_std"]
        err_out = (err_filled - stats["time_err_mean"]) / stats["time_err_std"]

    trk[:, IDX_TIME] = time_out
    trk[:, IDX_TIME_ERR] = err_out

    x[:n_trk] = trk
    data.x = x
    return data


def preprocess_trkst_features(data, stats):
    n_trk = data.n_trk
    x = data.x
    ts = x[n_trk:]
    if ts.shape[0] == 0:
        return data

    ts[:, IDX_TS_ETA] = (ts[:, IDX_TS_ETA] - stats["ts_eta_mean"]) / stats["ts_eta_std"]
    ts[:, IDX_TS_PHI] = (ts[:, IDX_TS_PHI] - stats["ts_phi_mean"]) / stats["ts_phi_std"]
    ts[:, IDX_TS_X] = (ts[:, IDX_TS_X] - stats["ts_x_mean"]) / stats["ts_x_std"]
    ts[:, IDX_TS_Y] = (ts[:, IDX_TS_Y] - stats["ts_y_mean"]) / stats["ts_y_std"]
    ts[:, IDX_TS_Z] = (ts[:, IDX_TS_Z] - stats["ts_z_mean"]) / stats["ts_z_std"]

    # dropped features -- zeroed out, kept in place for now
    ts[:, IDX_TS_PT] = 0.0
    ts[:, IDX_TS_EM_PT] = 0.0

    # raw_energy / raw_em_energy: log1p + standardize
    energy = torch.log1p(ts[:, IDX_TS_ENERGY])
    ts[:, IDX_TS_ENERGY] = (energy - stats["ts_energy_mean"]) / stats["ts_energy_std"]

    em_energy = torch.log1p(ts[:, IDX_TS_EM_ENERGY])
    ts[:, IDX_TS_EM_ENERGY] = (em_energy - stats["ts_em_energy_mean"]) / stats["ts_em_energy_std"]

    # time: clamp -> standardize, sentinel filled with TS_TIME_FILL
    time = ts[:, IDX_TS_TIME]
    err = ts[:, IDX_TS_TIME_ERR]
    mask = time != TS_TIME_SENTINEL

    time_clamped = torch.clamp(time, min=TS_TIME_CLAMP_MIN, max=TS_TIME_CLAMP_MAX)
    time_filled = torch.where(mask, time_clamped, torch.full_like(time, TS_TIME_FILL))
    time_out = (time_filled - stats["ts_time_mean"]) / stats["ts_time_std"]

    # timeError: sqrt real values -> standardize, sentinel filled with derived fill
    err_sqrt = torch.sqrt(torch.clamp(err, min=0.0))
    err_filled = torch.where(mask, err_sqrt, torch.full_like(err, stats["ts_time_err_fill"]))
    err_out = (err_filled - stats["ts_time_err_mean"]) / stats["ts_time_err_std"]

    ts[:, IDX_TS_TIME] = time_out
    ts[:, IDX_TS_TIME_ERR] = err_out

    x[n_trk:] = ts
    data.x = x
    return data


def preprocess_trk_ts_edge_features(data, stats):
    edge_attr = data.edge_attr
    mask = edge_attr[:, EDGE_IDX_TYPE] == 0
    n_edges = edge_attr.shape[0]

    # append the shared missing-time indicator column -- default 1.0 (valid)
    # for ts-ts edges, since their own corrected fill/indicator treatment is
    # not yet decided (TODO: revisit once ts-ts deltaTime/dtime_sig is redone)
    indicator = torch.ones(n_edges, 1, dtype=edge_attr.dtype)
    edge_attr = torch.cat([edge_attr, indicator], dim=1)

    e = edge_attr[mask]

    e[:, EDGE_IDX_DETA] = (e[:, EDGE_IDX_DETA] - stats["et_deta_mean"]) / stats["et_deta_std"]
    e[:, EDGE_IDX_DPHI] = (e[:, EDGE_IDX_DPHI] - stats["et_dphi_mean"]) / stats["et_dphi_std"]
    e[:, EDGE_IDX_DR] = (e[:, EDGE_IDX_DR] - stats["et_dR_mean"]) / stats["et_dR_std"]

    # dE -- dropped for trk-ts (redundant with E_ratio), zeroed out
    e[:, EDGE_IDX_DE] = 0.0

    log_E_ratio = torch.log10(e[:, EDGE_IDX_ERATIO])
    e[:, EDGE_IDX_ERATIO] = (log_E_ratio - stats["et_E_ratio_mean"]) / stats["et_E_ratio_std"]

    dt = e[:, EDGE_IDX_DT]
    dtsig = e[:, EDGE_IDX_DTSIG]
    dt_mask = dt != EDGE_DT_SENTINEL

    # Standardize and clip valid timing only. Missing timing keeps its
    # separately derived fill value and is identified by time_indicator.
    dt_out = torch.full_like(
        dt,
        (stats["et_dt_fill"] - stats["et_dt_mean"]) / stats["et_dt_std"],
    )
    dt_out[dt_mask] = torch.clamp(
        (dt[dt_mask] - stats["et_dt_mean"]) / stats["et_dt_std"],
        min=EDGE_DT_Z_CLIP_MIN,
        max=EDGE_DT_Z_CLIP_MAX,
    )
    e[:, EDGE_IDX_DT] = dt_out

    dtsig_log = torch.where(dt_mask, torch.log10(torch.clamp(dtsig, min=1e-8)),
                             torch.full_like(dtsig, stats["et_dtsig_fill"]))
    e[:, EDGE_IDX_DTSIG] = (dtsig_log - stats["et_dtsig_mean"]) / stats["et_dtsig_std"]

    e[:, EDGE_IDX_TIME_INDICATOR] = dt_mask.float()

    edge_attr[mask] = e
    data.edge_attr = edge_attr
    return data


def preprocess_ts_ts_edge_features(data, stats):
    edge_attr = data.edge_attr
    mask = edge_attr[:, EDGE_IDX_TYPE] == 1

    # append the shared indicator column if it doesn't exist yet (i.e. this
    # function is called before preprocess_trk_ts_edge_features) -- default
    # 1.0 (valid) for trk-ts rows until that function fills them in
    if edge_attr.shape[1] == EDGE_IDX_TIME_INDICATOR:
        indicator = torch.ones(edge_attr.shape[0], 1, dtype=edge_attr.dtype)
        edge_attr = torch.cat([edge_attr, indicator], dim=1)

    e = edge_attr[mask]

    e[:, EDGE_IDX_DETA] = (e[:, EDGE_IDX_DETA] - stats["ett_deta_mean"]) / stats["ett_deta_std"]
    e[:, EDGE_IDX_DPHI] = (e[:, EDGE_IDX_DPHI] - stats["ett_dphi_mean"]) / stats["ett_dphi_std"]
    e[:, EDGE_IDX_DR] = (e[:, EDGE_IDX_DR] - stats["ett_dR_mean"]) / stats["ett_dR_std"]

    log_dE = torch.sign(e[:, EDGE_IDX_DE]) * torch.log1p(torch.abs(e[:, EDGE_IDX_DE]))
    e[:, EDGE_IDX_DE] = (log_dE - stats["ett_dE_mean"]) / stats["ett_dE_std"]

    log_E_ratio = torch.log10(e[:, EDGE_IDX_ERATIO])
    e[:, EDGE_IDX_ERATIO] = (log_E_ratio - stats["ett_E_ratio_mean"]) / stats["ett_E_ratio_std"]

    dt = e[:, EDGE_IDX_DT]
    dtsig = e[:, EDGE_IDX_DTSIG]
    dt_mask = dt != EDGE_DT_SENTINEL

    dt_clipped = torch.clamp(dt, -TT_DT_CLIP_VALUE, TT_DT_CLIP_VALUE)
    dt_filled = torch.where(dt_mask, dt_clipped, torch.full_like(dt, stats["ett_dt_fill"]))
    e[:, EDGE_IDX_DT] = (dt_filled - stats["ett_dt_mean"]) / stats["ett_dt_std"]

    log_dtsig = torch.sign(dtsig) * torch.log1p(torch.abs(dtsig))
    dtsig_filled = torch.where(dt_mask, log_dtsig, torch.full_like(dtsig, stats["ett_dtsig_fill"]))
    e[:, EDGE_IDX_DTSIG] = (dtsig_filled - stats["ett_dtsig_mean"]) / stats["ett_dtsig_std"]

    e[:, EDGE_IDX_TIME_INDICATOR] = dt_mask.float()

    edge_attr[mask] = e
    data.edge_attr = edge_attr
    return data
