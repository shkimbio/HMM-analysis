# -*- coding: utf-8 -*-
"""fip_plots.py — figures for the FIP-HMM exception analysis (김수현)."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import fip_pipeline_복사본 as fp  # reuse loaders + config

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110})

ARCH = {  # archetype -> base color
    "Baseline": "#9AA3AF", "Elevated": "#D7263D", "Rising-transient": "#1F9E89",
    "Falling/offset": "#3457D5", "Suppressed": "#6A1B9A",
}


def scolor(label):
    base = label.split("-")[0] if not label.startswith("Falling") else "Falling/offset"
    if label.startswith("Rising"):
        base = "Rising-transient"
    return ARCH.get(base, "#777777")


def _save(fig, figdir, name, formats):
    for ext in formats:
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)


def main():
    cfg = fp.load_cfg()
    td = os.path.join(fp.ROOT, cfg["paths"]["tables_dir"])
    fd = os.path.join(fp.ROOT, cfg["paths"]["figures_dir"])
    os.makedirs(fd, exist_ok=True)
    fmts = cfg["figure_style"]["formats"]
    base = os.path.join(fp.ROOT, cfg["paths"]["base"])
    cpal = cfg["figure_style"]["channel_palette"]
    fs   = cfg["acquisition"]["fs_fip"]

    summary_threat, summary_shelter = [], []

    for m in cfg["mice"]:
        mid = m["id"]; mouse_dir = os.path.join(base, m["subdir"])
        lj = os.path.join(td, f"labels_{mid}.json")
        if not os.path.exists(lj):
            print("skip", mid, "(no labels)"); continue
        meta = json.load(open(lj, encoding="utf-8"))
        labels = {int(k): v for k, v in meta["labels"].items()}
        primary = m["primary"]; channels = m["channels"]
        states_df = pd.read_csv(os.path.join(td, f"states_{mid}.csv"))
        fr = pd.read_parquet(os.path.join(td, f"frames_{mid}.parquet"))
        ev = fp.load_events(mouse_dir)
        thr = ev.loc[ev["event"] == "Threat", "start_s"].values
        shl = fp.shelter_entry_times(mouse_dir, m["shelter_entry_source"], ev)
        rem = ev.loc[ev["event"] == "Shelter Remove", "start_s"].values
        title = f"{mid} · {m['sensor']}"

        # ---- FIG 0: HMM overview — ΔBIC · transition matrix · occupancy ----
        ks_df = pd.read_csv(os.path.join(td, f"ksel_{mid}.csv"))
        K_sel = len(states_df)
        k_vals = ks_df["K"].values
        bic_vals = ks_df["bic"].values
        bic_base = bic_vals[0]               # ΔBIC reference = smallest K in range
        delta_bic = bic_vals - bic_base
        transmat_path = os.path.join(td, f"transmat_{mid}.csv")
        tmat_df = pd.read_csv(transmat_path, index_col=0)
        tmat = tmat_df.values

        fig0, ax0s = plt.subplots(1, 3, figsize=(13, 4),
                                  gridspec_kw={"width_ratios": [1.5, 1.5, 1]})
        # panel a: ΔBIC
        ax0s[0].plot(k_vals, delta_bic, color="black", marker="o", lw=1.5, ms=5)
        k_sel_idx = np.where(k_vals == K_sel)[0]
        if len(k_sel_idx):
            ax0s[0].scatter([K_sel], [delta_bic[k_sel_idx[0]]], color="red", s=80,
                            zorder=5, label=f"Selected K={K_sel}")
            ax0s[0].legend(fontsize=9, frameon=False)
        ax0s[0].set_xlabel("# states (K)")
        ax0s[0].set_ylabel(f"ΔBIC (from K={k_vals[0]})")
        ax0s[0].set_title("a", loc="left", fontweight="bold")

        # panel b: transition matrix heatmap
        im = ax0s[1].imshow(tmat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        slabels = [f"S{i+1}" for i in range(K_sel)]
        for i in range(K_sel):
            for j in range(K_sel):
                ax0s[1].text(j, i, f"{tmat[i, j]:.2f}", ha="center", va="center",
                             fontsize=9,
                             color="white" if tmat[i, j] > 0.5 else "black")
        ax0s[1].set_xticks(range(K_sel)); ax0s[1].set_yticks(range(K_sel))
        ax0s[1].set_xticklabels(slabels); ax0s[1].set_yticklabels(slabels)
        ax0s[1].set_xlabel("State t"); ax0s[1].set_ylabel("State t-1")
        plt.colorbar(im, ax=ax0s[1], label="P(transition)", shrink=0.8)
        ax0s[1].set_title("b", loc="left", fontweight="bold")

        # panel c: occupancy bars
        occ_vals = states_df["occupancy"].values
        occ_colors = [scolor(l) for l in states_df["label"].values]
        ax0s[2].bar(range(K_sel), occ_vals, color=occ_colors)
        for i, v in enumerate(occ_vals):
            ax0s[2].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax0s[2].set_xticks(range(K_sel))
        ax0s[2].set_xticklabels(slabels, fontsize=9)
        ax0s[2].set_ylabel("Frac. occupancy")
        ax0s[2].set_ylim(0, max(occ_vals) * 1.3)
        ax0s[2].set_title("c", loc="left", fontweight="bold")

        fig0.suptitle(f"HMM Analysis  |  Mouse {mid} [{m['sensor']}]  |  K={K_sel} states",
                      fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save(fig0, fd, f"fig_{mid}_hmm_overview", fmts)

        # ---- FIG A: trace coloured by state — one subplot per channel ----
        t = fr["t_s"].values
        if len(thr):
            c = thr[0]; w0, w1 = c - 12, c + 30
        else:
            w0, w1 = t[0], t[0] + 60
        sel = (t >= w0) & (t <= w1)
        idx = np.where(sel)[0]
        nch = len(channels)
        fig, axes = plt.subplots(nch, 1, figsize=(11, 3.2 * nch), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]                    # flatten to 1-D
        for ci, (cname, col) in enumerate(channels.items()):
            ax = axes[ci]
            if len(idx):
                a = idx[0]
                for j in range(idx[0] + 1, idx[-1] + 2):
                    if j > idx[-1] or fr["label"].values[j] != fr["label"].values[a]:
                        ax.axvspan(t[a], t[min(j, len(t) - 1)],
                                   color=scolor(fr["label"].values[a]), alpha=0.25, lw=0)
                        a = j
            ax.plot(t[sel], fr[cname].values[sel], color="black", lw=0.8)
            for x in thr:
                if w0 <= x <= w1:
                    ax.axvline(x, color=cfg["figure_style"]["event_palette"]["Threat"],
                               lw=1.6, ls="--")
            for x in shl:
                if w0 <= x <= w1:
                    ax.axvline(x,
                               color=cfg["figure_style"]["event_palette"]["Shelter Return"],
                               lw=1.4, ls=":")
            ax.set_ylabel(f"{cname} ΔF/F (%)")
            if ci == 0:
                handles = [Patch(color=scolor(labels[k]), alpha=0.4, label=labels[k])
                           for k in sorted(labels)]
                ax.legend(handles=handles, fontsize=8, ncol=len(labels),
                          loc="upper right", frameon=False)
                ax.set_title(f"{title} — HMM state segmentation of ΔF/F"
                             "  (dashed=threat, dotted=shelter entry)")
        axes[-1].set_xlim(w0, w1)
        axes[-1].set_xlabel("time (s)")
        plt.tight_layout()
        _save(fig, fd, f"fig_{mid}_trace", fmts)

        # ---- FIG B: per-state mean dFF + occupancy ----
        K = len(states_df)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1]})
        xpos = np.arange(K)
        width = 0.8 / max(len(channels), 1)
        for ci, (cname, col) in enumerate(channels.items()):
            vals = states_df[f"{cname}_meandff"].values
            axes[0].bar(xpos + ci * width, vals, width, label=cname, color=cpal.get(cname, "#555"))
        axes[0].axhline(0, color="k", lw=0.6)
        axes[0].set_xticks(xpos + width * (len(channels) - 1) / 2)
        axes[0].set_xticklabels(states_df["label"], rotation=20, ha="right", fontsize=9)
        axes[0].set_ylabel("mean ΔF/F (%)"); axes[0].legend(fontsize=8, frameon=False)
        axes[0].set_title("per-state mean activity")
        axes[1].bar(xpos, states_df["occupancy"].values, color=[scolor(l) for l in states_df["label"]])
        axes[1].set_xticks(xpos); axes[1].set_xticklabels(states_df["label"], rotation=20, ha="right", fontsize=9)
        axes[1].set_ylabel("occupancy"); axes[1].set_title("state occupancy")
        fig.suptitle(title, y=1.02, fontsize=12, fontweight="bold")
        _save(fig, fd, f"fig_{mid}_states", fmts)

        # ---- FIG C: peri-event (threat + shelter) ----
        def peri_panel(axP, axD_list, stem, evt_color, evt_label, dur=None):
            ps = os.path.join(td, f"peri_{stem}_states_{mid}.csv")
            df = os.path.join(td, f"peri_{stem}_dff_{mid}.csv")
            if not (os.path.exists(ps) and os.path.getsize(ps) > 5):
                axP.text(0.5, 0.5, "n/a", ha="center"); return
            P = pd.read_csv(ps); D = pd.read_csv(df)
            if not len(P):
                axP.text(0.5, 0.5, "n/a", ha="center"); return
            for lab, g in P.groupby("state_label"):
                axP.plot(g["t_s"], g["mean_post"], color=scolor(lab), label=lab, lw=1.8)
                axP.fill_between(g["t_s"], g["mean_post"] - g["sem_post"], g["mean_post"] + g["sem_post"],
                                 color=scolor(lab), alpha=0.15, lw=0)
            axP.axvline(0, color=evt_color, lw=1.5, ls="--")
            if dur:
                axP.axvspan(0, dur, color=evt_color, alpha=0.08, lw=0)
            axP.set_ylabel("P(state)"); axP.set_ylim(0, 1)
            axP.set_title(f"{evt_label}: state probability"); axP.legend(fontsize=7, frameon=False, ncol=2)
            for (cname, col), axD in zip(channels.items(), axD_list):
                g = D[D["channel"] == col]
                axD.plot(g["t_s"], g["mean_dff"], color=cpal.get(cname, "#555"), label=cname, lw=1.8)
                axD.fill_between(g["t_s"], g["mean_dff"] - g["sem_dff"], g["mean_dff"] + g["sem_dff"],
                                 color=cpal.get(cname, "#555"), alpha=0.15, lw=0)
                axD.axvline(0, color=evt_color, lw=1.5, ls="--")
                if dur:
                    axD.axvspan(0, dur, color=evt_color, alpha=0.08, lw=0)
                axD.set_ylabel("ΔF/F (%)"); axD.legend(fontsize=7, frameon=False)
                axD.set_title(f"{evt_label}: {cname}")
            axD_list[-1].set_xlabel("time from event (s)")

        nch = len(channels)
        nrows = 1 + nch
        fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.2 * nrows), sharex="col")
        peri_panel(axes[0, 0], [axes[r, 0] for r in range(1, nrows)], "threat",
                   cfg["figure_style"]["event_palette"]["Threat"], "Threat onset", dur=cfg["events"]["threat_dur_s"])
        peri_panel(axes[0, 1], [axes[r, 1] for r in range(1, nrows)], "shelter",
                   cfg["figure_style"]["event_palette"]["Shelter Return"], "Shelter entry")
        fig.suptitle(f"{title} — event-aligned latent states & activity", y=1.0, fontsize=12, fontweight="bold")
        fig.tight_layout()
        _save(fig, fd, f"fig_{mid}_peri", fmts)

        # =========================================================
        # NEW FIGURES: K-sweep · session traces · 2D state space
        # =========================================================

        # Re-load raw dFF and rebuild features (same as pipeline)
        d_raw, _ = fp.load_dff(mouse_dir)
        primary_ch = {primary: channels[primary]}
        X_feat, feat_names_all = fp.build_features(
            d_raw, primary_ch, cfg["features"], fs)

        K_sel = len(states_df)
        k_list = cfg["hmm"]["k_range"]
        tab10 = plt.cm.tab10
        CHUNK = 300.0

        # Re-fit selected-K with same seeds -> reproduces pipeline model
        model_sel, _ = fp.fit_best(X_feat, K_sel, cfg["hmm"])
        post_sel     = model_sel.predict_proba(X_feat)
        states_sel   = model_sel.predict(X_feat)
        labels_refit = fp.label_states(model_sel, feat_names_all, primary)

        # S1=lowest level, SK=highest level
        li_idx  = feat_names_all.index(f"{primary}_level")
        sort_ord = list(np.argsort(model_sel.means_[:, li_idx]))
        rank_of  = {k: r for r, k in enumerate(sort_ord)}
        sname_of = {k: f"S{r+1}" for k, r in rank_of.items()}
        scol_of  = {f"S{r+1}": tab10(r / max(K_sel - 1, 1)) for r in range(K_sel)}

        thr_col = cfg["figure_style"]["event_palette"]["Threat"]
        shl_col = cfg["figure_style"]["event_palette"]["Shelter Return"]

        # chunk edges
        t_max = t[-1]
        edges = list(np.arange(0, t_max, CHUNK)) + [t_max]
        n_ch  = len(edges) - 1

        st_snames = np.array([sname_of[s] for s in states_sel])

        def _run_spans(t_arr, state_arr):
            if len(t_arr) == 0:
                return
            prev = state_arr[0]; a = 0
            for j in range(1, len(t_arr)):
                if state_arr[j] != prev:
                    yield t_arr[a], t_arr[j], prev
                    a = j; prev = state_arr[j]
            yield t_arr[a], t_arr[-1], prev

        def _shade_states(ax, tc, st_chunk):
            for ta, tb, sn in _run_spans(tc, st_chunk):
                ax.axvspan(ta, tb, color=scol_of[sn], alpha=0.25, lw=0)

        def _mark_events(ax, t0c, t1c, dur):
            for et in thr:
                if t0c <= et <= t1c:
                    ax.axvspan(et, min(et + dur, t1c), color=thr_col, alpha=0.18, lw=0)
            for et in shl:
                if t0c <= et <= t1c:
                    ax.axvspan(et, min(et + 3.0, t1c), color=shl_col, alpha=0.15, lw=0)

        def _peri_post(model_k, X_k, t_k, event_ts, win):
            post_k = model_k.predict_proba(X_k)
            pre_n  = int(round(-win[0] * fs)); post_n = int(round(win[1] * fs))
            rel    = np.arange(-pre_n, post_n + 1) / fs
            trials = []
            for et in event_ts:
                i0 = int(np.searchsorted(t_k, et))
                a, b = i0 - pre_n, i0 + post_n + 1
                if 0 <= a and b <= len(t_k):
                    trials.append(post_k[a:b])
            if not trials:
                return None, rel
            return np.nanmean(np.array(trials), axis=0), rel

        # ---- FIG: K sweep ----
        n_k = len(k_list)
        fig_sw, axs_sw = plt.subplots(
            n_k, 3, figsize=(15, 3.5 * n_k),
            gridspec_kw={"width_ratios": [3, 1.5, 1.5]})
        if n_k == 1:
            axs_sw = axs_sw[np.newaxis, :]

        for ri, K_k in enumerate(k_list):
            m_k, _ = fp.fit_best(X_feat, K_k, cfg["hmm"])
            if m_k is None:
                continue
            st_k  = m_k.predict(X_feat)
            li_k  = feat_names_all.index(f"{primary}_level")
            ord_k = list(np.argsort(m_k.means_[:, li_k]))
            rk_k  = {k: r for r, k in enumerate(ord_k)}

            # Viterbi sequence bar
            ax = axs_sw[ri, 0]
            ax.imshow(np.array([[rk_k[s] for s in st_k]]), aspect="auto",
                      cmap=plt.cm.get_cmap("tab10", K_k),
                      vmin=-0.5, vmax=K_k - 0.5,
                      extent=[t[0], t[-1], 0, 1])
            ax.set_yticks([])
            ax.set_ylabel(f"K={K_k}", rotation=0, labelpad=32, va="center")
            if ri == 0:
                ax.set_title("Viterbi state sequence", fontsize=9)
            if ri == n_k - 1:
                ax.set_xlabel("Time (s)", fontsize=8)
            ax.legend(handles=[Patch(color=tab10(r / max(K_k-1,1)), label=f"S{r+1}")
                                for r in range(K_k)],
                      fontsize=6, ncol=K_k, loc="upper right",
                      handlelength=1, frameon=True)

            # Peri-event panels
            for ci_p, (event_arr, win, ec, col_title) in enumerate([
                    (thr, cfg["events"]["threat_window"], thr_col, "Peri-threat  p(state)"),
                    (shl, cfg["events"]["shelter_window"], shl_col, "Peri-shelter entry  p(state)")]):
                ax_p = axs_sw[ri, 1 + ci_p]
                if len(event_arr):
                    mp, rel = _peri_post(m_k, X_feat, t, event_arr, win)
                    if mp is not None:
                        for r, ko in enumerate(ord_k):
                            ax_p.plot(rel, mp[:, ko],
                                      color=tab10(r / max(K_k-1,1)), lw=1.4,
                                      label=f"S{r+1}")
                ax_p.axvline(0, color=ec, lw=1.2, ls="--")
                ax_p.set_ylim(0, 1); ax_p.set_ylabel("p(state)", fontsize=7)
                if ri == 0:
                    ax_p.set_title(col_title, fontsize=9)
                    if ci_p == 0:
                        ax_p.legend(fontsize=6, frameon=False, ncol=2)
                if ri == n_k - 1:
                    ax_p.set_xlabel(
                        "Time from threat (s)" if ci_p == 0 else "Time from shelter (s)",
                        fontsize=8)

        fig_sw.suptitle(
            f"Mouse {mid} [{m['sensor']}] — K sweep (K={k_list[0]}→{k_list[-1]})",
            fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save(fig_sw, fd, f"fig_{mid}_ksweep", fmts)

        # ---- FIG: session P(state) ----
        fig_sp, axs_sp = plt.subplots(n_ch, 1, figsize=(13, 2.8 * n_ch))
        if n_ch == 1:
            axs_sp = [axs_sp]

        for ci in range(n_ch):
            t0c, t1c = edges[ci], edges[ci + 1]
            msk = (t >= t0c) & (t <= t1c)
            tc  = t[msk]; ax = axs_sp[ci]
            for r, ko in enumerate(sort_ord):
                sn = f"S{r+1}"
                ax.plot(tc, post_sel[msk, ko], color=scol_of[sn], lw=1.2, label=sn)
            _mark_events(ax, t0c, t1c, cfg["events"]["threat_dur_s"])
            ax.set_xlim(t0c, t1c); ax.set_ylim(0, 1.05)
            ax.set_ylabel("p(state)", fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.text(0.005, 0.93, f"{int(t0c)}→{int(t1c)} s",
                    transform=ax.transAxes, fontsize=7, va="top")
            if ci == 0:
                ax.legend(fontsize=8, frameon=False, ncol=K_sel, loc="upper right")

        fig_sp.suptitle(
            f"Mouse {mid} [{m['sensor']}]  |  K={K_sel} — "
            f"Session trace ({int(CHUNK)}s chunks)",
            fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save(fig_sp, fd, f"fig_{mid}_session_post", fmts)

        # ---- FIG: dFF + state background ----
        ch_list  = list(channels.items())
        nch_plot = len(ch_list)

        def _zsc(arr):
            return (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-9)

        z_data = {cn: _zsc(d_raw[col].values) for cn, col in ch_list}

        fig_dff, axs_dff = plt.subplots(
            n_ch * nch_plot, 1, figsize=(13, 2.2 * n_ch * nch_plot))
        if n_ch * nch_plot == 1:
            axs_dff = [axs_dff]

        for ci in range(n_ch):
            t0c, t1c = edges[ci], edges[ci + 1]
            msk = (t >= t0c) & (t <= t1c)
            tc = t[msk]; st_chunk = st_snames[msk]
            for chi, (cname, _col) in enumerate(ch_list):
                ax = axs_dff[ci * nch_plot + chi]
                _shade_states(ax, tc, st_chunk)
                ax.plot(tc, z_data[cname][msk], color=cpal.get(cname, "black"), lw=0.7)
                for et in thr:
                    if t0c <= et <= t1c:
                        ax.axvline(et, color=thr_col, lw=1.3, ls="--")
                for et in shl:
                    if t0c <= et <= t1c:
                        ax.axvline(et, color=shl_col, lw=1.1, ls=":")
                ax.set_xlim(t0c, t1c)
                ax.set_ylabel(f"{cname}\nΔF/F (z)", fontsize=7)
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.text(0.005, 0.93, f"{int(t0c)}→{int(t1c)} s",
                        transform=ax.transAxes, fontsize=7, va="top")
                if ci == 0 and chi == 0:
                    s_h = [Patch(color=scol_of[f"S{r+1}"], alpha=0.4, label=f"S{r+1}")
                           for r in range(K_sel)]
                    s_h += [Patch(color=thr_col, alpha=0.5, label="Threat"),
                            Patch(color=shl_col, alpha=0.5, label="Shelter")]
                    ax.legend(handles=s_h, fontsize=7,
                              ncol=K_sel + 2, loc="upper right", frameon=False)

        fig_dff.suptitle(
            f"Mouse {mid} [{m['sensor']}]  |  K={K_sel} — "
            f"ΔF/F 세션 트레이스 ({int(CHUNK)}s 청크)",
            fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save(fig_dff, fd, f"fig_{mid}_session_dff", fmts)

        # ---- FIG: dual channel overlay (multi-channel only) ----
        if nch_plot >= 2:
            ch1n, _ = ch_list[0]; ch2n, _ = ch_list[1]
            fig_ov, axs_ov = plt.subplots(n_ch, 1, figsize=(13, 2.8 * n_ch))
            if n_ch == 1:
                axs_ov = [axs_ov]
            for ci in range(n_ch):
                t0c, t1c = edges[ci], edges[ci + 1]
                msk = (t >= t0c) & (t <= t1c)
                tc = t[msk]; st_chunk = st_snames[msk]
                ax = axs_ov[ci]; ax2 = ax.twinx()
                _shade_states(ax, tc, st_chunk)
                l1, = ax.plot(tc, z_data[ch1n][msk],
                              color=cpal.get(ch1n, "#1F9E89"), lw=0.9,
                              label=f"{ch1n} (470c)")
                l2, = ax2.plot(tc, z_data[ch2n][msk],
                               color=cpal.get(ch2n, "#E5392F"), lw=0.9,
                               ls="--", label=f"{ch2n} (565nm)")
                ax.set_ylabel(f"{ch1n} (z)", fontsize=7,
                              color=cpal.get(ch1n, "#1F9E89"))
                ax2.set_ylabel(f"{ch2n} (z)", fontsize=7,
                               color=cpal.get(ch2n, "#E5392F"))
                ax.set_xlim(t0c, t1c)
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.text(0.005, 0.93, f"{int(t0c)}→{int(t1c)} s",
                        transform=ax.transAxes, fontsize=7, va="top")
                if ci == 0:
                    s_h = [Patch(color=scol_of[f"S{r+1}"], alpha=0.35, label=f"S{r+1}")
                           for r in range(K_sel)]
                    ax.legend(handles=[l1, l2] + s_h, fontsize=7,
                              ncol=2 + K_sel, loc="upper right", frameon=True)
            fig_ov.suptitle(
                f"Mouse {mid} [{m['sensor']}]  |  K={K_sel} — "
                f"듀얼 채널 오버레이 ({int(CHUNK)}s 청크)",
                fontsize=12, fontweight="bold")
            plt.tight_layout()
            _save(fig_ov, fd, f"fig_{mid}_session_overlay", fmts)

        # ---- FIG: 2D state space (primary level vs slope) ----
        li1 = feat_names_all.index(f"{primary}_level")
        li2_slope = f"{primary}_slope"
        if li2_slope in feat_names_all:
            li2 = feat_names_all.index(li2_slope)
            x_sc = X_feat[:, li1]; y_sc = X_feat[:, li2]

            sub   = max(1, len(x_sc) // 3000)
            xs    = x_sc[::sub]; ys = y_sc[::sub]
            c_arr = [scol_of[sname_of[s]] for s in states_sel[::sub]]

            fig_2d, (ax_sc2, ax_bar) = plt.subplots(
                1, 2, figsize=(11, 4.5),
                gridspec_kw={"width_ratios": [1.3, 1]})

            ax_sc2.scatter(xs, ys, c=c_arr, s=8, alpha=0.3, lw=0)
            for r, ko in enumerate(sort_ord):
                ax_sc2.scatter([model_sel.means_[ko, li1]],
                               [model_sel.means_[ko, li2]],
                               c=[scol_of[f"S{r+1}"]], s=130,
                               edgecolors="black", lw=1.5, zorder=5,
                               label=f"S{r+1}")
            ax_sc2.set_xlabel(f"{primary} level (z-score)", fontsize=9)
            ax_sc2.set_ylabel(f"{primary} slope (z-score)", fontsize=9)
            ax_sc2.legend(fontsize=8, frameon=False)
            ax_sc2.set_title(
                "2D State space (level vs slope)\n(□ = time sample,  ● = state mean)", fontsize=9)

            bar_w = 0.35; x_pos = np.arange(K_sel)
            lvl_means  = [model_sel.means_[sort_ord[r], li1] for r in range(K_sel)]
            slp_means  = [model_sel.means_[sort_ord[r], li2] for r in range(K_sel)]
            ax_bar.bar(x_pos - bar_w/2, lvl_means, bar_w,
                       color=cpal.get(primary, "#1F9E89"), label=f"{primary} level")
            ax_bar.bar(x_pos + bar_w/2, slp_means, bar_w,
                       color=cpal.get(primary, "#1F9E89"), alpha=0.5,
                       hatch="///", label=f"{primary} slope")
            ax_bar.axhline(0, color="k", lw=0.6)
            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels([f"S{r+1}" for r in range(K_sel)])
            ax_bar.set_ylabel("State mean (z-score)", fontsize=9)
            ax_bar.legend(fontsize=8, frameon=False)
            ax_bar.set_title("State별 평균 활성도", fontsize=9)

            fig_2d.suptitle(
                f"Mouse {mid} [{m['sensor']}]  |  K={K_sel} — 2D State space ({primary})",
                fontsize=12, fontweight="bold")
            plt.tight_layout()
            _save(fig_2d, fd, f"fig_{mid}_statespace", fmts)

        # ---- FIG: HMM state transition schematic ----
        from matplotlib.patches import Circle as _Circle

        def _draw_schematic(ax_sch):
            ax_sch.set_aspect("equal"); ax_sch.axis("off")
            R_lay  = 2.8    # layout circle radius
            R_node = 0.78   # state node radius
            thresh = 0.03   # min prob to draw arrow

            # Circular layout: S1 (lowest level) at top, clockwise
            angs = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, K_sel, endpoint=False)
            pos  = {f"S{r+1}": (R_lay * np.cos(angs[r]), R_lay * np.sin(angs[r]))
                    for r in range(K_sel)}
            li   = feat_names_all.index(f"{primary}_level")
            tmat = model_sel.transmat_

            # --- Arrows ---
            for i in range(K_sel):
                ki = sort_ord[i]; sni = f"S{i+1}"
                xi, yi = pos[sni]
                for j in range(K_sel):
                    kj = sort_ord[j]; snj = f"S{j+1}"
                    p  = tmat[ki, kj]
                    if p < thresh:
                        continue
                    xj, yj = pos[snj]
                    c_a = scol_of[sni]
                    if i == j:
                        # Self-loop on outer edge
                        lx = xi + (R_node + 0.38) * np.cos(angs[i])
                        ly = yi + (R_node + 0.38) * np.sin(angs[i])
                        ax_sch.add_patch(_Circle((lx, ly), 0.30, fill=False,
                                                  color=c_a, lw=1.8, zorder=1))
                        ax_sch.text(lx + 0.38 * np.cos(angs[i]),
                                    ly + 0.38 * np.sin(angs[i]),
                                    f"{p:.2f}", fontsize=7.5, ha="center",
                                    va="center", color=c_a, fontweight="bold",
                                    zorder=4)
                    else:
                        dx = xj - xi; dy = yj - yi
                        dist = np.hypot(dx, dy) + 1e-9
                        perp = np.array([-dy, dx]) / dist * 0.20
                        sx = xi + R_node * dx/dist + perp[0]
                        sy = yi + R_node * dy/dist + perp[1]
                        ex = xj - R_node * dx/dist + perp[0]
                        ey = yj - R_node * dy/dist + perp[1]
                        ax_sch.annotate("", xy=(ex, ey), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle="-|>", color=c_a,
                                            lw=1.6,
                                            connectionstyle="arc3,rad=0.15"),
                            zorder=1)
                        mx = (sx + ex)/2 + perp[0] * 1.0
                        my = (sy + ey)/2 + perp[1] * 1.0
                        ax_sch.text(mx, my, f"{p:.2f}", fontsize=7.5,
                                    ha="center", va="center", color=c_a,
                                    bbox=dict(fc="white", alpha=0.8, pad=1, lw=0),
                                    zorder=3)

            # --- State nodes ---
            for r in range(K_sel):
                sn = f"S{r+1}"; ko = sort_ord[r]
                x, y = pos[sn]; c = scol_of[sn]

                ax_sch.add_patch(_Circle((x, y), R_node, color=c, alpha=0.18, zorder=2))
                ax_sch.add_patch(_Circle((x, y), R_node, fill=False,
                                          color=c, lw=2.5, zorder=3))

                # S-label above node
                ax_sch.text(x, y + R_node + 0.24, sn, ha="center", va="bottom",
                            fontsize=11, fontweight="bold", color=c, zorder=5)

                # Human-readable label below node
                lbl = labels_refit.get(ko, sn)
                ax_sch.text(x, y - R_node - 0.14, lbl, ha="center", va="top",
                            fontsize=7.5, color="#222222", zorder=5)

                # Emission: Gaussian for primary level drawn inside node
                mu  = model_sel.means_[ko, li]
                _cov = np.asarray(model_sel.covars_[ko])
                sig = float(np.sqrt(_cov[li] if _cov.ndim == 1 else _cov[li, li]))
                sig = max(sig, 0.05)
                xg  = np.linspace(mu - 2.5*sig, mu + 2.5*sig, 60)
                yg  = np.exp(-0.5 * ((xg - mu)/sig)**2)
                w   = 0.55 * R_node
                h   = 0.42 * R_node
                xg_s = x + (xg - mu) / (2.5*sig) * w
                yg_s = y - 0.05 + yg * h
                ax_sch.fill_between(xg_s, y - 0.05, yg_s,
                                    color=c, alpha=0.55, zorder=4)
                ax_sch.plot(xg_s, yg_s, color=c, lw=1.2, zorder=4)
                ax_sch.text(x, y - 0.05 - h*0.05, f"μ={mu:.2f}",
                            ha="center", va="top", fontsize=6,
                            color="#555", zorder=5)

            pad = R_lay + R_node + 1.4
            ax_sch.set_xlim(-pad, pad); ax_sch.set_ylim(-pad, pad)

        fig_sc = plt.figure(figsize=(9, 9))
        _draw_schematic(fig_sc.add_subplot(111))
        fig_sc.suptitle(
            f"Mouse {mid} [{m['sensor']}]  |  K={K_sel} — "
            f"HMM State Transition Schematic",
            fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save(fig_sc, fd, f"fig_{mid}_schematic", fmts)

        # collect summary (primary-channel peri-threat/shelter, z within window)
        for stem, store in (("threat", summary_threat), ("shelter", summary_shelter)):
            dfp = os.path.join(td, f"peri_{stem}_dff_{mid}.csv")
            if os.path.exists(dfp) and os.path.getsize(dfp) > 5:
                D = pd.read_csv(dfp)
                g = D[D["channel"] == channels[primary]].copy()
                if len(g):
                    z = (g["mean_dff"] - g["mean_dff"].mean()) / (g["mean_dff"].std() or 1)
                    store.append((f"{mid} {m['sensor']}", g["t_s"].values, z.values, primary))

    # ---- SUMMARY figs: primary-channel z-scored response overlay ----
    for stem, store, evlab, dur in (("threat", summary_threat, "Threat onset", cfg["events"]["threat_dur_s"]),
                                     ("shelter", summary_shelter, "Shelter entry", None)):
        if not store:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for name, tt, z, prim in store:
            ax.plot(tt, z, lw=2.2, label=name, color=cpal.get(prim, None))
        ax.axvline(0, color="k", lw=1.4, ls="--")
        if dur:
            ax.axvspan(0, dur, color="#D7263D", alpha=0.07, lw=0)
        ax.set_xlabel("time from event (s)"); ax.set_ylabel("z-scored ΔF/F (within window)")
        ax.set_title(f"Cross-sensor {evlab}: primary-channel response (n=1 per sensor, descriptive)")
        ax.legend(fontsize=9, frameon=False)
        _save(fig, fd, f"fig_summary_{stem}", fmts)

    print("Saved figures ->", fd)


if __name__ == "__main__":
    main()
