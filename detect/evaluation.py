"""오탐(False Alarm) / 미탐(Miss Detection) 트레이드오프 정량화.

이 모듈이 팀원 C 파트의 핵심이다. 이상탐지는 "몇 건 잡았다"로는 평가할 수 없다.
문턱값을 낮추면 무조건 더 많이 잡히기 때문이다. 반드시 두 종류의 실패를 함께 봐야 한다:

    오탐 (False Alarm, FP) : 정상인데 경보    -> 현장이 경보를 무시하기 시작한다
    미탐 (Miss, FN)        : 이상인데 침묵    -> 불량 웨이퍼가 그대로 다음 공정으로 간다

반도체 FDC에서 문턱값 설정이 곧 이 둘의 균형을 어디에 둘 것인가의 문제이고,
그래서 라인마다 "ARL0는 최소 얼마 이상"처럼 수치 기준을 못박아 관리한다.

라벨 문제와 해법 — 합성 이상 주입(synthetic fault injection)
    실제 판매 데이터에는 '이 날이 진짜 이상이었다'는 정답 라벨이 없다.
    라벨이 없으면 오탐/미탐을 셀 수 없다. 그래서 정상으로 간주한 구간에
    크기와 시점을 아는 이상을 인위적으로 주입하고, 탐지기가 그걸 잡아내는지 측정한다.
    반도체에서도 신규 FDC 레시피를 검증할 때 과거 정상 로트에 가상 fault를 심어
    민감도를 확인하는 방식을 쓴다. 정답을 아는 문제를 만들어 채점하는 것이다.
"""

import numpy as np
import pandas as pd

from detect.control_charts import (
    shewhart_3sigma, ewma_chart, cusum_chart, isolation_forest_flags,
)

FAULT_TYPES = ("spike", "step", "drift")


# -----------------------------------------------------------------------------
# 1. 합성 이상 주입 — 정답(ground truth)을 아는 검증용 데이터 만들기
# -----------------------------------------------------------------------------

def inject_faults(series, fault_type="step", n_faults=5, magnitude=2.0,
                  duration=7, seed=42, margin=40):
    """정상 시계열에 크기·시점을 아는 이상을 주입하고 정답 라벨을 함께 반환한다.

    magnitude는 계열 표준편차의 배수다. magnitude=2.0이면 '2시그마짜리 이상'.

    fault_type
        spike : 하루만 튀는 단발 이상.        설비 순간 정지, 배송 사고
        step  : duration일 동안 평균이 이동.  레시피 변경, 경쟁점 개업
        drift : duration일에 걸쳐 서서히 증가. 챔버 오염 누적, 계절 상권 이동

    margin일 이전 구간은 관리도 워밍업(기준선 계산)에 필요하므로 주입하지 않는다.

    반환: (이상이 주입된 Series, 정답 라벨 Series[bool])
    """
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"fault_type은 {FAULT_TYPES} 중 하나여야 합니다: {fault_type}")

    rng = np.random.default_rng(seed)
    out = series.astype(float).copy()
    labels = pd.Series(False, index=series.index)

    sigma = float(series.std())
    if sigma == 0 or np.isnan(sigma):
        return out, labels

    shift = magnitude * sigma
    n = len(series)
    span = duration if fault_type != "spike" else 1

    # 이상 구간끼리 겹치지 않도록 뒤쪽 구간을 균등 분할해 배치한다.
    usable = n - margin - span
    if usable <= 0 or n_faults <= 0:
        return out, labels

    slot = usable // max(n_faults, 1)
    if slot < span:
        n_faults = max(1, usable // max(span, 1))
        slot = usable // max(n_faults, 1)

    for i in range(n_faults):
        lo = margin + i * slot
        hi = min(lo + max(slot - span, 1), n - span)
        if hi <= lo:
            start = lo
        else:
            start = int(rng.integers(lo, hi))
        end = start + span
        sign = 1.0 if rng.random() < 0.5 else -1.0

        if fault_type == "spike":
            out.iloc[start] += sign * shift
        elif fault_type == "step":
            out.iloc[start:end] += sign * shift
        else:  # drift — 0에서 shift까지 선형 증가
            ramp = np.linspace(0.0, shift, span)
            out.iloc[start:end] += sign * ramp

        labels.iloc[start:end] = True

    out = out.clip(lower=0)     # 판매량은 음수가 될 수 없다
    return out, labels


def _label_events(labels):
    """연속된 True 구간을 하나의 '이상 이벤트'로 묶어 [(시작idx, 끝idx), ...] 반환."""
    events, start = [], None
    vals = labels.values
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif not v and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(vals) - 1))
    return events


# -----------------------------------------------------------------------------
# 2. 혼동행렬 · 성능지표
# -----------------------------------------------------------------------------

def confusion_counts(labels, flags, tolerance=1):
    """정답 라벨과 탐지 결과로 혼동행렬을 센다.

    tolerance(일): 이상 발생 당일이 아니라 하루 이틀 뒤에 잡아도 탐지 성공으로 인정한다.
    관리도는 원래 며칠에 걸쳐 신호가 쌓이므로(특히 CUSUM), 정확히 당일만 인정하면
    실제로 잡은 것을 미탐으로 오판하게 된다. 이 완화가 없으면 CUSUM이 부당하게 불리하다.

    반환: dict(TP, FP, FN, TN)
        TP : 이상 근처에서 울린 경보 수
        FP : 정상 구간에서 울린 경보 수  -> 오탐
        FN : 끝내 못 잡은 이상일 수      -> 미탐
        TN : 조용했던 정상일 수
    """
    lab = labels.values.astype(bool)
    flg = flags.reindex(labels.index).fillna(False).values.astype(bool)
    n = len(lab)

    def _near(arr, i):
        lo, hi = max(0, i - tolerance), min(n, i + tolerance + 1)
        return arr[lo:hi].any()

    tp = sum(1 for i in range(n) if flg[i] and _near(lab, i))
    fp = sum(1 for i in range(n) if flg[i] and not _near(lab, i))
    fn = sum(1 for i in range(n) if lab[i] and not _near(flg, i))
    tn = n - tp - fp - fn
    return {"TP": tp, "FP": fp, "FN": fn, "TN": max(tn, 0)}


def average_run_length(labels, flags):
    """실측 ARL0 — 정상 구간에서 오경보 한 번이 뜨기까지의 평균 관측 수.

    ARL0 = 정상일 수 / 오경보 수

    클수록 좋다. 3시그마 관리도의 이론값은 약 370이다(정상 370일에 한 번 헛경보).
    실측값이 20 같은 숫자가 나오면 "3주에 한 번씩 헛경보가 울린다"는 뜻이고,
    현장에서는 그 경보를 곧 무시하게 된다. 오탐률보다 직관적이라 실무에서 이 지표를 쓴다.
    """
    lab = labels.values.astype(bool)
    flg = flags.reindex(labels.index).fillna(False).values.astype(bool)
    in_control = ~lab
    n_normal = int(in_control.sum())
    n_false = int((flg & in_control).sum())
    if n_false == 0:
        return float("inf")
    return n_normal / n_false


def detection_delay(labels, flags):
    """이상 이벤트별 탐지 지연(일)의 평균. 못 잡은 이벤트는 제외하고 평균낸다.

    반도체에서 이 지연이 곧 '불량이 몇 장 더 흘러갔는가'다. 지연 1일과 5일은
    같은 탐지율이어도 손실 규모가 전혀 다르므로 반드시 따로 본다.
    """
    flg = flags.reindex(labels.index).fillna(False).values.astype(bool)
    delays = []
    for start, end in _label_events(labels):
        hit = np.where(flg[start:end + 1])[0]
        if len(hit) > 0:
            delays.append(int(hit[0]))
    if not delays:
        return None
    return float(np.mean(delays))


def detector_metrics(labels, flags, tolerance=1):
    """오탐/미탐 트레이드오프를 한 줄로 요약하는 지표 묶음."""
    cm = confusion_counts(labels, flags, tolerance)
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    far = fp / (fp + tn) if (fp + tn) else 0.0    # 오탐률 (정상일 중 헛경보 비율)
    mdr = fn / (fn + tp) if (fn + tp) else 0.0    # 미탐률 (이상일 중 놓친 비율)

    events = _label_events(labels)
    flg = flags.reindex(labels.index).fillna(False)
    caught = sum(1 for s, e in events if flg.values[s:e + 1].any())

    return {
        **cm,
        "정밀도": round(precision, 4),
        "재현율": round(recall, 4),
        "F1": round(f1, 4),
        "오탐률(FAR)": round(far, 4),
        "미탐률(MDR)": round(mdr, 4),
        "실측ARL0": average_run_length(labels, flags),
        "평균탐지지연(일)": detection_delay(labels, flags),
        "이벤트탐지율": round(caught / len(events), 4) if events else 0.0,
        "이벤트수": len(events),
        "경보수": int(flg.sum()),
    }


# -----------------------------------------------------------------------------
# 3. 문턱값 스윕 — 트레이드오프 곡선 그리기
# -----------------------------------------------------------------------------

# 탐지기별로 '문턱값 손잡이'가 무엇인지 등록해 둔다.
DETECTOR_KNOBS = {
    "3시그마": {
        "func": lambda s, v: shewhart_3sigma(s, L=v)["is_anomaly"],
        "param": "L (시그마 배수)",
        "values": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "default": 3.0,
    },
    "EWMA": {
        "func": lambda s, v: ewma_chart(s, lam=0.2, L=v)["is_anomaly"],
        "param": "L (시그마 배수)",
        "values": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "default": 3.0,
    },
    "CUSUM": {
        "func": lambda s, v: cusum_chart(s, k=0.5, h=v)["is_anomaly"],
        "param": "h (결정구간)",
        "values": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
        "default": 5.0,
    },
    "IsolationForest": {
        "func": lambda s, v: isolation_forest_flags(s, contamination=v),
        "param": "contamination (이상 비율)",
        "values": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
        "default": 0.05,
    },
}


def threshold_sweep(series, labels, detector="3시그마", values=None, tolerance=1):
    """문턱값을 바꿔가며 오탐률과 미탐률이 어떻게 맞바뀌는지 표로 만든다.

    이 표의 요점은 "최적값 하나"가 아니라 어떤 값을 골라도 둘 다 좋아지지는 않는다는
    사실 자체다. 문턱을 조이면 오탐이 줄고 미탐이 늘며, 풀면 반대다.
    어디를 고를지는 통계가 아니라 비용이 정한다 —
    헛경보 한 번의 비용과 놓친 불량 한 장의 비용 중 무엇이 비싼가.
    """
    if detector not in DETECTOR_KNOBS:
        raise ValueError(f"알 수 없는 탐지기: {detector}")
    knob = DETECTOR_KNOBS[detector]
    vals = values if values is not None else knob["values"]

    rows = []
    for v in vals:
        flags = knob["func"](series, v)
        m = detector_metrics(labels, flags, tolerance)
        arl = m["실측ARL0"]
        rows.append({
            "탐지기": detector,
            knob["param"]: v,
            "오탐률(FAR)": m["오탐률(FAR)"],
            "미탐률(MDR)": m["미탐률(MDR)"],
            "정밀도": m["정밀도"],
            "재현율": m["재현율"],
            "F1": m["F1"],
            "실측ARL0": None if arl == float("inf") else round(arl, 1),
            "평균탐지지연(일)": m["평균탐지지연(일)"],
            "경보수": m["경보수"],
        })
    return pd.DataFrame(rows)


def benchmark_detectors(series, fault_type="step", n_faults=5, magnitude=2.0,
                        duration=7, seed=42, tolerance=1, params=None):
    """관리도 3종 + Isolation Forest를 같은 이상 시나리오로 채점해 한 표에 비교한다.

    반환: (성능 비교 DataFrame, 이상 주입된 Series, 정답 라벨 Series)

    fault_type을 바꿔가며 돌려보면 각 관리도의 성격이 숫자로 드러난다.
    spike에서는 3시그마가, drift에서는 CUSUM이 이기는 게 정상이다.
    한 관리도가 모든 이상 유형에서 이기는 일은 없고, 그래서 여러 개를 함께 쓴다.
    """
    faulty, labels = inject_faults(
        series, fault_type=fault_type, n_faults=n_faults,
        magnitude=magnitude, duration=duration, seed=seed,
    )
    p = params or {}
    runs = {
        "3시그마": shewhart_3sigma(faulty, L=p.get("L", 3.0))["is_anomaly"],
        "EWMA": ewma_chart(faulty, lam=p.get("lam", 0.2), L=p.get("ewma_L", 3.0))["is_anomaly"],
        "CUSUM": cusum_chart(faulty, k=p.get("k", 0.5), h=p.get("h", 5.0))["is_anomaly"],
        "IsolationForest": isolation_forest_flags(faulty, contamination=p.get("contamination", 0.05)),
    }
    # 다수결 합의도 하나의 '탐지기'로 함께 채점한다.
    vote = pd.DataFrame(runs).sum(axis=1) >= p.get("vote_threshold", 2)
    runs["합의(2개+)"] = vote

    rows = []
    for name, flags in runs.items():
        m = detector_metrics(labels, flags, tolerance)
        arl = m["실측ARL0"]
        rows.append({
            "탐지기": name,
            "오탐(FP)": m["FP"],
            "미탐(FN)": m["FN"],
            "적중(TP)": m["TP"],
            "오탐률(FAR)": m["오탐률(FAR)"],
            "미탐률(MDR)": m["미탐률(MDR)"],
            "정밀도": m["정밀도"],
            "재현율": m["재현율"],
            "F1": m["F1"],
            "이벤트탐지율": m["이벤트탐지율"],
            "평균탐지지연(일)": m["평균탐지지연(일)"],
            "실측ARL0": None if arl == float("inf") else round(arl, 1),
        })
    return pd.DataFrame(rows), faulty, labels
