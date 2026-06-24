# -*- coding: utf-8 -*-
"""fip_report.py — markdown report (criteria + references + rationale + 해설)
for the FIP-HMM exception analysis (김수현 / Soohyun Kim)."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import fip_pipeline_복사본 as fp


def _t(td, n): return os.path.join(td, n)


def peak_after(td, mid, stem, col, lo=0.0, hi=5.0):
    p = _t(td, f"peri_{stem}_dff_{mid}.csv")
    if not (os.path.exists(p) and os.path.getsize(p) > 5):
        return None
    D = pd.read_csv(p); g = D[D["channel"] == col]
    if not len(g):
        return None
    w = g[(g["t_s"] >= lo) & (g["t_s"] <= hi)]
    base = g[g["t_s"] < 0]["mean_dff"].mean()
    if not len(w):
        return None
    imax = w["mean_dff"].abs().idxmax()
    return {"peak": float(w.loc[imax, "mean_dff"]), "t": float(w.loc[imax, "t_s"]),
            "baseline": float(base), "delta": float(w.loc[imax, "mean_dff"] - base)}


def dom_state_post(td, mid, stem, lo=0.0, hi=4.0):
    p = _t(td, f"peri_{stem}_states_{mid}.csv")
    if not (os.path.exists(p) and os.path.getsize(p) > 5):
        return None
    P = pd.read_csv(p); w = P[(P["t_s"] >= lo) & (P["t_s"] <= hi)]
    if not len(w):
        return None
    g = w.groupby("state_label")["mean_post"].mean().sort_values(ascending=False)
    return g.index[0], float(g.iloc[0])


def main():
    cfg = fp.load_cfg()
    td = os.path.join(fp.ROOT, cfg["paths"]["tables_dir"])
    rd = os.path.join(fp.ROOT, cfg["paths"]["reports_dir"])
    os.makedirs(rd, exist_ok=True)
    summ = pd.read_csv(_t(td, "summary.csv"))

    L = []
    L.append("# FIP-HMM 예외 분석 — 위협-유발 은신처 회피 중 잠재 신경활동 상태\n")
    L.append("**대상:** 김수현(Soohyun Kim) fiber-photometry 데이터 · 야생형 B6SJL 2마리 "
             "(센서별 1마리). **과제는 동일**(위협-유발 은신처 회피), 그러나 마우스 라인·기록 방식이 "
             "기존 5xFAD pose-HMM 작업과 달라 **독립된 예외 분석**으로 수행했고, 이후 5xFAD 분석/로스터에는 "
             "반영하지 않는다.\n")

    L.append("## 0. 한 줄 요약")
    L.append("> DLC pose/h5가 없으므로, HMM을 **광유전(FIP) ΔF/F 신호에 직접** 적용해 "
             "**잠재 신경활동 상태**를 추출하고, 이를 위협·은신처 이벤트에 정렬했다. "
             "센서가 서로 달라 **마우스별로 따로** 모형화했으며 **센서당 n=1 → 기술적 사례연구**다 "
             "(집단 통계 아님).\n")

    L.append("## 1. 데이터")
    L.append(summ.to_markdown(index=False))
    L.append("\n- ΔF/F는 제공된 FIP 파이프라인에서 **등흡광(415 nm) 보정**까지 끝난 값(`470nm_dFF_corrected`, "
             "`565nm_dFF`)을 사용. 표본화 20 Hz.")
    L.append("- 이벤트(Threat / Shelter Return / Shelter Remove)는 photometry 시계(초)에 기록되어 있어 "
             "ΔF/F와 직접 정렬 가능(별도 영상 동기화 불필요).")
    L.append("- 채널↔생물학: **dLight**=세포외 도파민, **jRGECO**=적색 Ca²⁺(신경활동), "
             "**pdyn-GCaMP**=dynorphin⁺(직접경로/D1형 MSN) Ca²⁺, **CaMKII-GCaMP**=흥분성(주세포) Ca²⁺. "
             "단일 GCaMP 마우스의 565 채널은 사실상 공백(대조)이라 제외.\n")

    L.append("## 2. 분석 기준·선정 이유·레퍼런스 (핵심 요청 항목)\n")
    L.append("### 2.1 왜 pose가 아니라 FIP 신호에 HMM을 적용했나")
    L.append("- 이 세션에는 DLC 결과(h5)도, 적용 가능한 학습 모델도 없고, top-camera 로그에는 위치 좌표가 없다 "
             "(프레임 타임스탬프만). 따라서 기존 pose 기반 5특징 HMM을 **그대로 재현할 수 없다.**")
    L.append("- 대신 **연속 신호를 임의 임계값 없이 재발하는 잠재 상태로 분해**한다는 HMM의 동일한 철학을, "
             "이번엔 **신경 신호(ΔF/F)** 에 적용했다. 광유전 신호는 본질적으로 시계열이므로 Gaussian HMM의 "
             "관측으로 자연스럽다 (Rabiner 1989; Wiltschko 2015; Markowitz 2018/2023).")
    L.append("### 2.2 광유전 전처리 기준")
    L.append("- **등흡광(415 nm) 회귀 보정 + ΔF/F**: 운동/혈류 아티팩트 제거의 표준 (Martianova 2019; Lerner 2015). "
             "제공 파이프라인 산출물을 그대로 채택(재처리하지 않음 — 보정본 신뢰).")
    L.append("- 특징화 전 **0.1–99.9 백분위 클립**으로 극단 아티팩트만 제거(생리적 transient는 보존).")
    L.append("### 2.3 HMM 관측 특징: 레벨 + 기울기")
    L.append("- 각 채널에서 **z-점수화한 ΔF/F(레벨)** 와 **그 시간미분(기울기)** 을 함께 사용. "
             "레벨만 쓰면 상태가 진폭(고/저)으로만 갈리지만, 기울기를 더하면 **상승 transient / 하강·소거 / "
             "지속 고활성 / 기저**를 구분할 수 있다 — 광유전 신호의 핵심 구조(phasic 이벤트)를 포착하기 위함.")
    L.append("- 레벨은 0.15 s, 기울기는 0.30 s 가우시안 평활(미분 잡음 안정화) 후 산출.")
    L.append("### 2.4 마우스별 개별 HMM (풀링하지 않음)")
    L.append("- 두 마우스는 **서로 다른 센서**(도파민 vs D1-MSN Ca)를 측정한다. "
             "신호의 '상태'가 생물학적으로 같은 대상이 아니므로 **풀링은 부적절**하다 → 각 마우스를 독립 모형화하고, "
             "**이벤트에 대한 반응 패턴**만 정성적으로 비교한다.")
    L.append("### 2.5 상태 개수 K 선택")
    L.append("- 마우스당 세션이 1개뿐이라 leave-one-mouse-out이 불가 → **세션 내 5-겹 연속 분할 교차검증**으로 "
             "held-out 로그우도를 최대화하는 K를 고르고 **BIC**를 병기(Schwarz 1978). K 후보 = {2,3,4,5}.")
    L.append("- 공분산 = 대각, EM 재시작 6회(최량 우도 채택), 시드 고정 — 기존 파이프라인과 동일한 보수적 설정.")
    L.append("### 2.6 이벤트 정렬")
    L.append("- 위협 개시·은신처 진입(10171은 trial별 추정치, 10102는 'Shelter Return' 이벤트) 전후로 "
             "**상태 사후확률 P(state)** 과 **평균 ΔF/F**를 정렬·평균. 은신처 제거(probe)는 1회뿐이라 "
             "전/후 기술 비교만.")
    L.append("### 2.7 센서 생물학 레퍼런스")
    L.append("- dLight: Patriarchi 2018 · jRGECO1a: Dana 2016 · GCaMP6: Chen 2013 · "
             "pdyn=직접경로(D1) 표지: Gerfen & Surmeier 2011, Kravitz 2010 · 은신처 회피: Evans 2018.\n")

    L.append("## 3. 마우스별 결과\n")
    for m in cfg["mice"]:
        mid = m["id"]; prim = m["primary"]; col = m["channels"][prim]
        sdf = pd.read_csv(_t(td, f"states_{mid}.csv"))
        ks = pd.read_csv(_t(td, f"ksel_{mid}.csv"))
        K = len(sdf)
        L.append(f"### {mid} · {m['sensor']}  (K={K})")
        L.append("상태 서명(레벨/기울기 z, 채널별 평균 ΔF/F, 점유율, 체류):\n")
        show = ["label", "occupancy", "dwell_s"] + [c for c in sdf.columns if c.endswith("_meandff")]
        L.append(sdf[show].round(3).to_markdown(index=False))
        thr_pk = peak_after(td, mid, "threat", col); sh_pk = peak_after(td, mid, "shelter", col)
        thr_ds = dom_state_post(td, mid, "threat"); sh_ds = dom_state_post(td, mid, "shelter")
        bullet = []
        if thr_pk:
            bullet.append(f"위협 후 0–5 s {prim} 변화 ≈ {thr_pk['delta']:+.2f}%p (peak {thr_pk['peak']:+.2f}% @ {thr_pk['t']:.1f}s)")
        if thr_ds:
            bullet.append(f"위협 직후(0–4 s) 우세 상태: **{thr_ds[0]}** (P≈{thr_ds[1]:.2f})")
        if sh_pk:
            bullet.append(f"은신처 진입 후 0–5 s {prim} 변화 ≈ {sh_pk['delta']:+.2f}%p")
        if sh_ds:
            bullet.append(f"은신처 진입 직후 우세 상태: **{sh_ds[0]}** (P≈{sh_ds[1]:.2f})")
        rp = _t(td, f"shelter_remove_{mid}.csv")
        if os.path.exists(rp):
            rdf = pd.read_csv(rp)
            pre = rdf[f"{prim}_pre"].iloc[0]; post = rdf[f"{prim}_post"].iloc[0]
            bullet.append(f"은신처 제거 전/후 평균 {prim} ΔF/F: {pre:+.2f}% → {post:+.2f}%")
        for b in bullet:
            L.append(f"- {b}")
        L.append(f"\n_K 선택_: " + " · ".join(
            f"K{int(r.K)} CV={r.cv_heldout_ll_per_sample:.3f}" for r in ks.itertuples()) + "\n")

    L.append("## 4. 센서 간 종합 (정성적, 센서당 n=1)")
    L.append("- 그림 `fig_summary_threat` / `fig_summary_shelter`는 각 마우스 **주 채널**의 위협·은신처 진입 "
             "전후 반응을 (창 내 z-점수로) 겹쳐 보여준다. 도파민(dLight)·직접경로 MSN(pdyn)이 "
             "같은 행동 사건에서 어떻게 다르게 움직이는지에 대한 **가설 생성용** 비교다.")
    L.append("- 해석은 전적으로 사례적이다: 센서당 1마리이므로 어떤 차이도 통계적 주장이 아니라 "
             "**후속 실험에서 검증할 관찰**로만 제시한다.\n")

    L.append("## 5. 한계와 다음 단계")
    L.append("- **센서당 n=1, 단일 세션** → 집단/유전형 통계 불가. HMM 상태는 *세션 내* 기술적 분해다.")
    L.append("- pose가 없어 행동 자체의 미세 상태(도피/동결 등)는 직접 분해하지 못함 → 행동은 이벤트 마커로만 대표.")
    L.append("- 기록 부위가 메타데이터에 없음(주 채널 해석 시 유의).")
    L.append("- **강화 방안**: (i) 동일 센서 마우스 추가로 집단화, (ii) 영상에 경량 추적(또는 DLC) 적용해 "
             "pose-HMM 상태와 FIP 상태를 교차정렬, (iii) 위협/은신처 이벤트별 transient 검정(부트스트랩).\n")

    L.append("## 6. 참고문헌")
    refs = [
        "Rabiner LR (1989) A tutorial on hidden Markov models and selected applications in speech recognition. Proc IEEE 77:257–286.",
        "Wiltschko AB et al. (2015) Mapping sub-second structure in mouse behavior. Neuron 88:1121–1135.",
        "Markowitz JE et al. (2018) The striatum organizes 3D behavior via moment-to-moment action selection. Cell 174:44–58.",
        "Markowitz JE et al. (2023) Spontaneous behaviour is structured by reinforcement without explicit reward. Nature 614:108–117.",
        "Calhoun AJ, Pillow JW, Murthy M (2019) Unsupervised identification of the internal states that shape natural behavior. Nat Neurosci 22:2040–2049.",
        "Ashwood ZC et al. (2022) Mice alternate between discrete strategies during perceptual decision-making. Nat Neurosci 25:201–212.",
        "Patriarchi T et al. (2018) Ultrafast neuronal imaging of dopamine dynamics with designed genetically encoded sensors. Science 360:eaat4422.",
        "Dana H et al. (2016) Sensitive red protein calcium indicators for imaging neural activity. eLife 5:e12727.",
        "Chen TW et al. (2013) Ultrasensitive fluorescent proteins for imaging neuronal activity. Nature 499:295–300.",
        "Martianova E, Aronson S, Proulx CD (2019) Multi-fibre photometry to record neural activity in freely-moving animals. J Vis Exp 152:e60278.",
        "Lerner TN et al. (2015) Intact-brain analyses reveal distinct information carried by SNc dopamine subcircuits. Cell 162:635–647.",
        "Gerfen CR, Surmeier DJ (2011) Modulation of striatal projection systems by dopamine. Annu Rev Neurosci 34:441–466.",
        "Kravitz AV et al. (2010) Regulation of parkinsonian motor behaviours by optogenetic control of basal ganglia circuitry. Nature 466:622–626.",
        "Evans DA et al. (2018) A synaptic threshold mechanism for computing escape decisions. Nature 558:590–594.",
        "Schwarz G (1978) Estimating the dimension of a model. Ann Stat 6:461–464.",
    ]
    for i, r in enumerate(refs, 1):
        L.append(f"{i}. {r}")
    L.append("\n_참고문헌은 방법론·기준의 근거로 제시한 것이며, 정확한 서지정보는 사용 전 확인 권장._")

    out = os.path.join(rd, "FIP_HMM_분석보고서_KR.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
