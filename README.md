### pipeline --> plots --> report 순서대로 실행.




# FIP-HMM Analysis

fiber photometry ΔF/F 신호에 Gaussian HMM을 적용해 잠재적 신경 활동 state를 추출하고,
위협(Threat) 및 은신처 진입(Shelter entry) 이벤트에 정렬한 분석 파이프라인.

> **n = 1 per sensor — 기술적 단일 피험체 케이스 스터디, 집단 통계 아님**

---

## 분석 대상 마우스

| Mouse ID | Sensor | Primary channel | Line |
|----------|--------|-----------------|------|
| 10171 | dLight + jRGECO | dLight (도파민) | B6SJL WT |
| 10102 | pdyn-GCaMP | pdyn-GCaMP (D1-MSN Ca²⁺) | B6SJL WT |

- **HMM은 primary channel만으로 학습** (state 정의는 dLight 또는 pdyn-GCaMP 단독)
- secondary channel (jRGECO)은 state별 반응 관찰 용도로만 사용

---

## 디렉토리 구조

```
김수현/
├── fip_config.yaml                  # 전체 설정 파일
├── src - 복사본/
│   ├── fip_pipeline_2026-06-24 ver.py   # HMM 학습 · 테이블 생성
│   └── fip_plots_2026-06-24 ver.py      # 피규어 생성
├── FIP_wt_dLight+jRGECO/10171/      # 10171 원본 데이터
├── FIP_pdyn_FGCaMP/10102/           # 10102 원본 데이터
└── results/
    ├── tables/                      # 파이프라인 출력 CSV/JSON/parquet
    └── figures/                     # 피규어 PNG + SVG
```

---

## 실행 환경

**Python**: `C:\Users\USER\miniconda3\python.exe`  
**필수 패키지**: `hmmlearn`, `numpy`, `pandas`, `scipy`, `matplotlib`, `pyarrow`, `pyyaml`

```powershell
# 패키지 확인
& "C:\Users\USER\miniconda3\python.exe" -c "import hmmlearn; print(hmmlearn.__version__)"
```

---

## 실행 방법

반드시 **pipeline → plots 순서**로 실행.

```powershell
# 1. HMM 학습 및 테이블 생성
& "C:\Users\USER\miniconda3\python.exe" "src - 복사본\fip_pipeline_2026-06-24 ver.py"

# 2. 피규어 생성
& "C:\Users\USER\miniconda3\python.exe" "src - 복사본\fip_plots_2026-06-24 ver.py"
```

작업 디렉토리는 `김수현/` 폴더.

---

## 주요 설정 (fip_config.yaml)

```yaml
hmm:
  k_range: [2, 3, 4, 5]   # BIC 계산 및 k-sweep 시각화 범위
  force_k: 3               # 실제 사용할 K 강제 고정 (CV 무시)
  covariance_type: "diag"
  n_iter: 200
  n_init: 6
  cv_folds: 5
```

- `force_k`를 제거하면 CV 기준으로 자동 K 선택
- `k_range`는 BIC overview 및 k-sweep 피규어에 항상 전체 범위 표시

---

## 출력 피규어 목록

### 마우스별 (`{mid}` = 10171 / 10102)

| 파일명 | 내용 |
|--------|------|
| `fig_{mid}_hmm_overview` | (a) ΔBIC curve (K=2~5, 선택된 K 빨간 점) · (b) 전이 행렬 · (c) state 점유율 |
| `fig_{mid}_trace` | ΔF/F 트레이스에 HMM state 색상 배경 오버레이 |
| `fig_{mid}_states` | state별 평균 ΔF/F 및 점유율 막대 |
| `fig_{mid}_peri` | 이벤트 정렬: state P(state) + 채널별 ΔF/F (채널 분리 패널) |
| `fig_{mid}_ksweep` | K=2~5 sweep: Viterbi 시퀀스 + peri-event P(state) |
| `fig_{mid}_session_post` | 세션 전체 P(state) 시계열 (300s 청크) |
| `fig_{mid}_session_dff` | 세션 전체 ΔF/F + state 배경 (300s 청크) |
| `fig_{mid}_session_overlay` | 듀얼 채널 오버레이 (10171만 해당) |
| `fig_{mid}_statespace` | 2D state space: primary level vs slope |
| `fig_{mid}_schematic` | HMM state 전이 다이어그램 |

### 크로스-센서 summary

| 파일명 | 내용 |
|--------|------|
| `fig_summary_threat` | 두 센서 primary channel z-scored ΔF/F 오버레이 (Threat) |
| `fig_summary_shelter` | 두 센서 primary channel z-scored ΔF/F 오버레이 (Shelter entry) |
| `fig_summary_threat_hmm` | 센서별 분리: ΔF/F + HMM P(state) (Threat) |
| `fig_summary_shelter_hmm` | 센서별 분리: ΔF/F + HMM P(state) (Shelter entry) |

---

## 분석 설계 메모

- **HMM feature**: primary channel의 `level` (smoothed ΔF/F) + `slope` (미분) — 각각 z-score
- **State 라벨링**: primary level 평균값 기준 → Suppressed / Baseline / Elevated 자동 분류
- **jRGECO 취급**: HMM 학습에서 제외. state 정의는 dLight만으로 수행하고 jRGECO는 각 state에서의 반응 관찰에만 사용 (두 센서를 합치면 생물학적 의미가 모호해짐)
- **이벤트**: Threat onset (위협 노출 시작) / Shelter entry (은신처 진입 시각)
  - 10171: shelter entry는 `threat_shelter_response_estimates.csv` 기반 추정값
  - 10102: shelter entry는 `timeline_events.csv`의 "Shelter Return" 이벤트

