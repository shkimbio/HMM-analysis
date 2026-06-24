# -*- coding: utf-8 -*-
"""fip_pipeline.py — EXCEPTION analysis (Soohyun Kim / 김수현 FIP dataset).

Fit a Gaussian HMM DIRECTLY on fiber-photometry ΔF/F (no DLC pose exists for
these sessions) to extract latent neural-activity states, then align those
states to threat and shelter events. One HMM per mouse (sensors differ).

n = 1 per sensor -> descriptive single-subject case studies, NOT group stats.
"""
from __future__ import annotations
import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))   # Desktop/김수현


def load_cfg():
    with open(os.path.join(HERE, "..", "fip_config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def find1(*pattern_parts):
    hits = glob.glob(os.path.join(*pattern_parts), recursive=True)
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
#  I/O                                                                        #
# --------------------------------------------------------------------------- #
def load_dff(mouse_dir):
    p = find1(mouse_dir, "**", "Raw_dFF_Data_ROI0.csv")
    d = pd.read_csv(p)
    d = d.rename(columns={"Time (s)": "t_s"})
    return d, p


def _session_t0(mouse_dir):
    """Anchor for hardware-clock event files: the GUI 'start' timestamp, which
    matches the 0-based 'elapsed' convention used by the other sessions and the
    experimenter's own curated alignment. Falls back to the first raw FP sample.
    (GUI-start vs FP-start differ by ~1-2 s -> documented alignment precision.)"""
    st = find1(mouse_dir, "**", "session_timestamps.csv")
    if st:
        s = pd.read_csv(st)
        row = s[s["event"].astype(str).str.strip() == "start"]
        if len(row):
            return float(row["timestamp_s"].iloc[0])
    raw = find1(mouse_dir, "**", "*_470nm_raw_*.csv")
    if raw:
        return float(pd.read_csv(raw, header=None, nrows=1).iloc[0, 0])
    return 0.0


def load_events(mouse_dir):
    p = find1(mouse_dir, "**", "timeline_events.csv")
    e = pd.read_csv(p)
    e.columns = [c.strip() for c in e.columns]
    if "elapsed_ms" in e.columns:
        start = e["elapsed_ms"].astype(float) / 1000.0
        end = e["end_ms"].astype(float) / 1000.0
    elif "start_timestamp_s" in e.columns:
        # hardware clock -> shift to the 0-based session/dFF clock
        t0 = _session_t0(mouse_dir)
        start = e["start_timestamp_s"].astype(float) - t0
        end = e["end_timestamp_s"].astype(float) - t0
    else:
        raise ValueError(f"unknown event schema: {e.columns.tolist()}")
    name = e["event"].astype(str).str.strip()
    return pd.DataFrame({"event": name, "start_s": start, "end_s": end})


def shelter_entry_times(mouse_dir, source, ev):
    if source == "estimates":
        p = find1(mouse_dir, "**", "threat_shelter_response_estimates.csv")
        if p:
            s = pd.read_csv(p)
            return s["shelter_entry_s"].astype(float).dropna().values
        return np.array([])
    return ev.loc[ev["event"] == "Shelter Return", "start_s"].values


# --------------------------------------------------------------------------- #
#  Features                                                                   #
# --------------------------------------------------------------------------- #
def _z(x):
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if sd > 1e-9 else 1.0)


def build_features(d, channels, fcfg, fs):
    lo, hi = fcfg["clip_pct"]
    cols, names = [], []
    for cname, col in channels.items():
        x = d[col].values.astype(float)
        x = np.clip(x, np.nanpercentile(x, lo), np.nanpercentile(x, hi))
        lvl = gaussian_filter1d(x, max(fcfg["level_smooth_s"] * fs, 0.5))
        cols.append(_z(lvl)); names.append(f"{cname}_level")
        if fcfg.get("use_slope", True):
            sm = gaussian_filter1d(x, max(fcfg["slope_smooth_s"] * fs, 0.5))
            slope = np.gradient(sm) * fs
            cols.append(_z(slope)); names.append(f"{cname}_slope")
    return np.column_stack(cols).astype(float), names


# --------------------------------------------------------------------------- #
#  HMM fit + within-session CV for K                                          #
# --------------------------------------------------------------------------- #
def _fit_once(X, K, cov, n_iter, seed, lengths=None):
    m = GaussianHMM(n_components=K, covariance_type=cov, n_iter=n_iter,
                    random_state=seed, min_covar=1e-3, tol=1e-3)
    m.fit(X, lengths)
    return m


def fit_best(X, K, hc, lengths=None):
    best, best_ll = None, -np.inf
    for i in range(hc["n_init"]):
        try:
            m = _fit_once(X, K, hc["covariance_type"], hc["n_iter"], hc["seed"] + i, lengths)
            ll = m.score(X, lengths)
            if ll > best_ll:
                best, best_ll = m, ll
        except Exception:
            continue
    return best, best_ll


def cv_select(X, hc):
    n = len(X); folds = hc["cv_folds"]
    edges = np.linspace(0, n, folds + 1).astype(int)
    blocks = [(edges[i], edges[i + 1]) for i in range(folds)]
    rows = []
    for K in hc["k_range"]:
        hll = []
        for ti, (a, b) in enumerate(blocks):
            Xtest = X[a:b]
            train_parts, lengths = [], []
            for tj, (c, d) in enumerate(blocks):
                if tj == ti:
                    continue
                train_parts.append(X[c:d]); lengths.append(d - c)
            Xtr = np.concatenate(train_parts, axis=0)
            try:
                m, _ = fit_best(Xtr, K, {**hc, "n_init": max(2, hc["n_init"] // 2)}, lengths)
                hll.append(m.score(Xtest) / len(Xtest))
            except Exception:
                hll.append(np.nan)
        # full-data BIC
        mf, llf = fit_best(X, K, hc)
        nfeat = X.shape[1]
        nparams = (K - 1) + K * (K - 1) + K * nfeat + K * nfeat   # init+trans+means+diag covars
        bic = -2 * llf + nparams * np.log(n)
        rows.append({"K": K, "cv_heldout_ll_per_sample": np.nanmean(hll),
                     "cv_sd": np.nanstd(hll), "full_ll": llf, "bic": bic})
    cv = pd.DataFrame(rows)
    bestK = int(hc["force_k"]) if hc.get("force_k") else int(cv.loc[cv["cv_heldout_ll_per_sample"].idxmax(), "K"])
    return cv, bestK


# --------------------------------------------------------------------------- #
#  State labelling (data-driven, from primary-channel level/slope)            #
# --------------------------------------------------------------------------- #
def label_states(model, names, primary):
    li = names.index(f"{primary}_level")
    si = names.index(f"{primary}_slope") if f"{primary}_slope" in names else None
    labels = {}
    for k in range(model.n_components):
        lvl = model.means_[k, li]
        slp = model.means_[k, si] if si is not None else 0.0
        if si is not None and slp > 0.35 and lvl > -0.2:
            base = "Rising-transient"
        elif si is not None and slp < -0.35 and lvl < 0.2:
            base = "Falling/offset"
        elif lvl > 0.45:
            base = "Elevated"
        elif lvl < -0.45:
            base = "Suppressed"
        else:
            base = "Baseline"
        labels[k] = base
    # disambiguate duplicates by level rank
    seen = {}
    order = np.argsort([model.means_[k, li] for k in range(model.n_components)])
    for k in labels:
        seen.setdefault(labels[k], []).append(k)
    for base, ks in seen.items():
        if len(ks) > 1:
            for rank, k in enumerate(sorted(ks, key=lambda kk: model.means_[kk, li])):
                labels[k] = f"{base}-{rank+1}"
    return labels


# --------------------------------------------------------------------------- #
#  Event-aligned posteriors + dFF                                             #
# --------------------------------------------------------------------------- #
def peri(event_t, t, post, dff_cols, d, win, fs, labels):
    pre, postw = win
    n_pre, n_post = int(round(-pre * fs)), int(round(postw * fs))
    rel = np.arange(-n_pre, n_post + 1) / fs
    K = post.shape[1]
    P, D = [], {c: [] for c in dff_cols}
    for et in event_t:
        i0 = int(np.searchsorted(t, et))
        a, b = i0 - n_pre, i0 + n_post + 1
        if a < 0 or b > len(t):
            continue
        P.append(post[a:b, :])
        for c in dff_cols:
            D[c].append(d[c].values[a:b])
    if not P:
        return pd.DataFrame(), pd.DataFrame(), 0
    P = np.array(P)                       # (nev, T, K)
    rows = []
    for k in range(K):
        mp = np.nanmean(P[:, :, k], axis=0); sp = np.nanstd(P[:, :, k], axis=0) / np.sqrt(P.shape[0])
        for j, tt in enumerate(rel):
            rows.append({"t_s": tt, "state_label": labels[k], "mean_post": mp[j], "sem_post": sp[j]})
    drows = []
    for c in dff_cols:
        arr = np.array(D[c]); mc = np.nanmean(arr, axis=0); sc = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
        for j, tt in enumerate(rel):
            drows.append({"t_s": tt, "channel": c, "mean_dff": mc[j], "sem_dff": sc[j]})
    return pd.DataFrame(rows), pd.DataFrame(drows), P.shape[0]


def dwell_occ(states, K, fs):
    occ = np.array([(states == k).mean() for k in range(K)])
    dwell = []
    for k in range(K):
        runs, c = [], 0
        for s in states:
            if s == k:
                c += 1
            elif c:
                runs.append(c); c = 0
        if c:
            runs.append(c)
        dwell.append(np.mean(runs) / fs if runs else 0.0)
    return occ, np.array(dwell)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    cfg = load_cfg()
    fs = cfg["acquisition"]["fs_fip"]
    td = os.path.join(ROOT, cfg["paths"]["tables_dir"])
    os.makedirs(td, exist_ok=True)
    base = os.path.join(ROOT, cfg["paths"]["base"])
    summary = []

    for m in cfg["mice"]:
        mid = m["id"]; mouse_dir = os.path.join(base, m["subdir"])
        print("=" * 64); print(f"{mid}  [{m['sensor']}]")
        d, dff_p = load_dff(mouse_dir)
        ev = load_events(mouse_dir)
        t = d["t_s"].values
        channels = m["channels"]
        dff_cols = list(channels.values())

        primary_ch = {m["primary"]: channels[m["primary"]]}
        X, names = build_features(d, primary_ch, cfg["features"], fs)
        cv, K = cv_select(X, cfg["hmm"])
        cv.to_csv(os.path.join(td, f"ksel_{mid}.csv"), index=False)
        print(cv.round(3).to_string(index=False)); print(f"  selected K={K}")

        model, ll = fit_best(X, K, cfg["hmm"])
        states = model.predict(X)
        post = model.predict_proba(X)
        labels = label_states(model, names, m["primary"])

        # order states by primary level for readability
        li = names.index(f"{m['primary']}_level")
        order = list(np.argsort(model.means_[:, li]))
        occ, dwell = dwell_occ(states, K, fs)

        srow = []
        for k in range(K):
            row = {"state": k, "label": labels[k], "occupancy": occ[k], "dwell_s": dwell[k]}
            for fi, nm in enumerate(names):
                row[nm + "_z"] = model.means_[k, fi]
            for cname, col in channels.items():
                row[f"{cname}_meandff"] = float(np.nanmean(d[col].values[states == k])) if (states == k).any() else np.nan
            srow.append(row)
        states_df = pd.DataFrame(srow).iloc[order].reset_index(drop=True)
        states_df.to_csv(os.path.join(td, f"states_{mid}.csv"), index=False)
        print(states_df.round(2).to_string(index=False))

        pd.DataFrame(model.transmat_, index=[labels[k] for k in range(K)],
                     columns=[labels[k] for k in range(K)]).to_csv(os.path.join(td, f"transmat_{mid}.csv"))

        # frames for plotting
        fr = pd.DataFrame({"t_s": t, "state": states, "label": [labels[s] for s in states]})
        for cname, col in channels.items():
            fr[cname] = d[col].values
        fr.to_parquet(os.path.join(td, f"frames_{mid}.parquet"))

        # event-aligned
        thr = ev.loc[ev["event"] == "Threat", "start_s"].values
        ps, pd_, nthr = peri(thr, t, post, dff_cols, d, cfg["events"]["threat_window"], fs, labels)
        ps.to_csv(os.path.join(td, f"peri_threat_states_{mid}.csv"), index=False)
        pd_.to_csv(os.path.join(td, f"peri_threat_dff_{mid}.csv"), index=False)

        sh = shelter_entry_times(mouse_dir, m["shelter_entry_source"], ev)
        ss, sd_, nsh = peri(sh, t, post, dff_cols, d, cfg["events"]["shelter_window"], fs, labels)
        ss.to_csv(os.path.join(td, f"peri_shelter_states_{mid}.csv"), index=False)
        sd_.to_csv(os.path.join(td, f"peri_shelter_dff_{mid}.csv"), index=False)

        # shelter-remove probe (pre vs post), if present
        rem = ev.loc[ev["event"] == "Shelter Remove", "start_s"].values
        if len(rem):
            rt = rem[0]; pre = states[t < rt]; postm = states[t >= rt]
            rrow = []
            for k in range(K):
                rrow.append({"state": k, "label": labels[k],
                             "occ_pre": float((pre == k).mean()) if len(pre) else np.nan,
                             "occ_post": float((postm == k).mean()) if len(postm) else np.nan})
            rdf = pd.DataFrame(rrow)
            for cname, col in channels.items():
                rdf[f"{cname}_pre"] = float(np.nanmean(d[col].values[t < rt]))
                rdf[f"{cname}_post"] = float(np.nanmean(d[col].values[t >= rt]))
            rdf["remove_t_s"] = rt
            rdf.to_csv(os.path.join(td, f"shelter_remove_{mid}.csv"), index=False)

        with open(os.path.join(td, f"labels_{mid}.json"), "w", encoding="utf-8") as f:
            json.dump({"id": mid, "sensor": m["sensor"], "K": K,
                       "labels": {str(k): labels[k] for k in labels},
                       "feature_names": names, "order": [int(o) for o in order]}, f,
                      ensure_ascii=False, indent=2)

        summary.append({"id": mid, "sensor": m["sensor"], "line": m["line"],
                        "n_samples": len(d), "dur_min": round(t[-1] / 60, 1),
                        "fs_Hz": fs, "K": K, "n_threat": int(len(thr)),
                        "n_shelter_entry": int(len(sh)), "shelter_remove": int(len(rem) > 0),
                        "primary_channel": m["primary"]})

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(td, "summary.csv"), index=False)
    print("=" * 64); print(sdf.to_string(index=False))
    print("\nWrote tables ->", td)


if __name__ == "__main__":
    main()
