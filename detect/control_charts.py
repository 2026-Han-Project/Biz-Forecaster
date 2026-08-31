"""통계적 공정관리(SPC) 관리도 3종 + 머신러닝 이상탐지 1종.

반도체 공정의 FDC(Fault Detection & Classification)에서 쓰는 관리도를
빵집 일별 판매량 시계열에 그대로 적용한다. 대상이 웨이퍼냐 빵이냐만 다르고
'정상 범위를 통계적으로 정의하고, 벗어나면 경보를 울린다'는 구조는 동일하다.

세 관리도는 서로 다른 이상 유형에 강하다 (이게 3종을 다 구현하는 이유다):

    Shewhart 3σ : 단발성 급변(spike)에 강함.   점진적 드리프트는 거의 못 잡는다.
    EWMA        : 완만한 평균 이동에 강함.      과거를 지수가중으로 기억한다.
    CUSUM       : 작은 편차의 누적에 가장 강함. 챔버 오염처럼 서서히 쌓이는 이상용.

공통 설계 원칙 — 셀프마스킹(self-masking) 방지
    당일 값을 자기 관리한계 계산에 넣으면, 이상치가 자기 기준선을 끌어올려
    스스로를 정상으로 만들어 버린다. 그래서 모든 관리한계는 series.shift(1),
    즉 '전날까지의 값'만으로 계산한다.
"""

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# 교과서 기준 ARL0(관리상태 평균런길이) — 이상이 없을 때 오경보가 뜨기까지의 평균 관측 수.
# 클수록 오탐이 적다. 실측 ARL0와 비교하면 파라미터가 적절한지 판단할 수 있다.
NOMINAL_ARL0 = {
    "3시그마": 370.4,   # 정규분포에서 |z|>3 확률 0.0027 -> 1/0.0027
    "EWMA": 500.0,      # lam=0.2, L=3.0 기준 근사값
    "CUSUM": 465.0,     # k=0.5, h=5.0 기준 표준값
}


def _prior_baseline(series, window=28, min_periods=10):
    """전날까지의 값으로만 계산한 (중심선, 표준편차). 셀프마스킹 방지의 핵심."""
    prior = series.shift(1)
    center = prior.rolling(window, min_periods=min_periods).mean()
    sigma = prior.rolling(window, min_periods=min_periods).std()
    return center, sigma


def shewhart_3sigma(series, L=3.0, window=28, min_periods=10, non_negative=True):
    """Shewhart 개별값 관리도 (통칭 3시그마 관리도).

    관리한계: center +- L * sigma
    L을 낮추면 민감해지지만(미탐 감소) 오탐이 급증한다. 이 L이 바로
    evaluation.threshold_sweep()에서 스윕하는 트레이드오프 손잡이다.
    """
    center, sigma = _prior_baseline(series, window, min_periods)
    ucl = center + L * sigma
    lcl = center - L * sigma
    if non_negative:
        # 원계열(판매량)은 음수가 될 수 없으므로 하한을 0에서 자른다.
        # 단 잔차 관리도에서는 잔차가 0을 중심으로 오르내리므로 이 클립을 끄지 않으면
        # 음수 잔차가 전부 하한 이탈로 잡혀 오탐이 폭증한다.
        lcl = lcl.clip(lower=0)
    is_anomaly = ((series > ucl) | (series < lcl)).fillna(False)
    return pd.DataFrame({
        "value": series, "center": center, "ucl": ucl, "lcl": lcl,
        "is_anomaly": is_anomaly,
    })


def ewma_chart(series, lam=0.2, L=3.0, window=28, min_periods=10, non_negative=True):
    """EWMA(지수가중이동평균) 관리도.

        z_t = lam * x_t + (1 - lam) * z_{t-1}

    lam이 작을수록 과거를 길게 기억해 완만한 이동에 민감해지고, 단발 스파이크에는 둔해진다.
    관리한계 폭은 sqrt(lam / (2 - lam))만큼 좁아진다 (EWMA의 분산이 원계열보다 작기 때문).
    """
    center, sigma = _prior_baseline(series, window, min_periods)
    z = series.ewm(alpha=lam, adjust=False).mean()
    half_width = L * sigma * np.sqrt(lam / (2.0 - lam))
    ucl = center + half_width
    lcl = center - half_width
    if non_negative:
        lcl = lcl.clip(lower=0)
    is_anomaly = ((z > ucl) | (z < lcl)).fillna(False)
    return pd.DataFrame({
        "value": series, "ewma": z, "center": center, "ucl": ucl, "lcl": lcl,
        "is_anomaly": is_anomaly,
    })


def cusum_chart(series, k=0.5, h=5.0, window=28, min_periods=10):
    """표형(tabular) CUSUM 관리도. — 팀원 C 파트에서 새로 추가한 세 번째 관리도.

    표준화한 편차 z_t = (x_t - center_t) / sigma_t 를 누적한다:

        C+_t = max(0, C+_{t-1} + z_t - k)      상향 이탈 누적
        C-_t = max(0, C-_{t-1} - z_t - k)      하향 이탈 누적
        경보  : C+ > h  또는  C- > h

    k(reference value)는 '무시할 편차'의 크기. 통상 탐지하려는 이동량의 절반으로 잡는다.
    k=0.5면 1시그마 이동을 표적으로 삼는다는 뜻이다.
    h(decision interval)는 경보 문턱. h를 키우면 오탐이 줄고 탐지가 늦어진다.

    3시그마가 '한 방에 크게 튀어야' 잡는 반면, CUSUM은 0.5시그마씩 며칠 연속 밀리는
    작은 편차도 누적해서 잡아낸다. 반도체에서 챔버 오염·타겟 소모처럼 서서히 진행되는
    드리프트를 잡는 데 CUSUM을 쓰는 이유가 이것이다.

    경보 후에는 C+/C-를 0으로 리셋한다(재시작). 리셋하지 않으면 한 번 넘긴 뒤
    계속 경보 상태로 남아 이상 '구간'을 셀 수 없다.
    """
    center, sigma = _prior_baseline(series, window, min_periods)
    z = (series - center) / sigma.replace(0, np.nan)

    c_pos = np.zeros(len(series))
    c_neg = np.zeros(len(series))
    flags = np.zeros(len(series), dtype=bool)

    prev_pos = prev_neg = 0.0
    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            # 기준선을 아직 못 만든 워밍업 구간 — 누적하지 않고 통과시킨다.
            c_pos[i] = c_neg[i] = 0.0
            continue
        cur_pos = max(0.0, prev_pos + zi - k)
        cur_neg = max(0.0, prev_neg - zi - k)
        c_pos[i], c_neg[i] = cur_pos, cur_neg

        if cur_pos > h or cur_neg > h:
            flags[i] = True
            prev_pos = prev_neg = 0.0    # 경보 후 리셋
        else:
            prev_pos, prev_neg = cur_pos, cur_neg

    return pd.DataFrame({
        "value": series,
        "cusum_pos": pd.Series(c_pos, index=series.index),
        "cusum_neg": pd.Series(c_neg, index=series.index),
        "h": h,
        "center": center,
        "is_anomaly": pd.Series(flags, index=series.index),
    })


def isolation_forest_flags(series, contamination=0.05, random_state=42):
    """Isolation Forest — 값·요일·7일이동평균 대비 편차를 피처로 쓰는 비지도 이상탐지.

    관리도와 결정적으로 다른 점: contamination 비율만큼을 '항상' 이상으로 뽑는다.
    즉 정상만 있는 구간에서도 5%는 무조건 경보가 뜬다. 절대적 관리한계가 없고
    상대적 순위만 매기기 때문이다. 이 성질이 오탐률에 그대로 드러난다.
    """
    if not _SKLEARN_OK or len(series) < 30:
        return pd.Series(False, index=series.index)

    weekday = series.index.dayofweek.values
    roll7 = series.rolling(7, min_periods=1).mean().values
    features = np.column_stack([series.values, weekday, series.values - roll7])

    model = IsolationForest(contamination=contamination, random_state=random_state)
    pred = model.fit_predict(features)
    return pd.Series(pred == -1, index=series.index)


def run_all_detectors(series, L=3.0, lam=0.2, ewma_L=3.0,
                      k=0.5, h=5.0, contamination=0.05, vote_threshold=2,
                      residual_mode="none", n_weeks=4):
    """관리도 3종 + Isolation Forest를 한 번에 돌리고 다수결로 합의한다.

    반환: (탐지기별 상세 DataFrame dict, 합의 결과 DataFrame)

    vote_threshold개 이상의 방법이 동의할 때만 최종 '이상'으로 판정한다.
    단독 판정보다 오탐이 크게 줄어드는데, 그 대가로 미탐이 늘어난다.
    이 트레이드오프도 evaluation.benchmark_detectors()에서 수치로 확인할 수 있다.
    """
    # residual_mode='weekday'면 요일 프로파일을 제거한 잔차를 관리도에 올린다.
    # 잔차는 0을 중심으로 오르내리므로 하한 클립(non_negative)을 반드시 꺼야 한다.
    charted, expected = residualize(series, mode=residual_mode, n_weeks=n_weeks)
    if residual_mode != "none":
        charted = charted.fillna(0.0)
    non_neg = (residual_mode == "none")

    sigma_df = shewhart_3sigma(charted, L=L, non_negative=non_neg)
    ewma_df = ewma_chart(charted, lam=lam, L=ewma_L, non_negative=non_neg)
    cusum_df = cusum_chart(charted, k=k, h=h)
    # Isolation Forest는 요일을 이미 피처로 쓰므로 항상 원계열에 적용한다.
    iso_flags = isolation_forest_flags(series, contamination=contamination)

    combined = pd.DataFrame({
        "3시그마": sigma_df["is_anomaly"],
        "EWMA": ewma_df["is_anomaly"],
        "CUSUM": cusum_df["is_anomaly"],
        "IsolationForest": iso_flags,
    })
    combined["탐지방법수"] = combined.sum(axis=1)
    combined["이상여부"] = combined["탐지방법수"] >= vote_threshold

    details = {
        "3시그마": sigma_df,
        "EWMA": ewma_df,
        "CUSUM": cusum_df,
        "IsolationForest": pd.DataFrame({"value": series, "is_anomaly": iso_flags}),
    }
    details["_charted"] = charted        # 관리도에 실제로 올라간 계열 (원계열 또는 잔차)
    details["_expected"] = expected      # 요일 기대값 (residual_mode='none'이면 None)
    return details, combined


# -----------------------------------------------------------------------------
# 잔차 관리도 — 관리도의 전제(정상상태)를 맞춰주는 전처리
# -----------------------------------------------------------------------------

def weekday_residual(series, n_weeks=4, min_periods=2):
    """요일 프로파일을 제거한 잔차를 만든다.

    왜 필요한가.
        관리도는 '공정이 정상상태(in-control)일 때 값이 하나의 분포를 따른다'고 전제한다.
        그런데 빵집 판매량은 주말에 뛰고 평일에 가라앉는 주기가 있어, 이 전제가 깨진다.
        이때 표준편차가 '요일 간 차이'까지 흡수해 부풀고, 관리한계가 그만큼 넓어져
        진짜 이상을 놓치게 된다(미탐 증가). 실제로 이 프로젝트 데이터에서
        요일 성분을 남겨두면 2.5시그마 이상의 탐지율이 20%까지 떨어졌다.

    해법.
        각 날짜에 대해 '직전 n_weeks주간 같은 요일의 평균'을 기대값으로 잡고 빼준다.
        남은 잔차는 요일 효과가 제거되어 정상상태 가정에 훨씬 가까워진다.
        반도체 FDC에서 레시피 스텝별 기대 프로파일을 뺀 잔차를 관리도에 올리는 것과
        같은 처리다. 원신호가 아니라 '설명되지 않는 부분'을 감시하는 것이다.

    기대값도 shift(1)을 거쳐 과거 데이터로만 계산하므로 셀프마스킹이 생기지 않는다.
    """
    by_weekday = series.groupby(series.index.dayofweek)
    expected = by_weekday.transform(
        lambda x: x.shift(1).rolling(n_weeks, min_periods=min_periods).mean()
    )
    residual = series - expected
    return residual, expected


def residualize(series, mode="weekday", n_weeks=4):
    """관리도에 올릴 계열을 고른다. mode='none'이면 원계열 그대로.

    반환: (관리도에 올릴 계열, 기대값 계열 또는 None)
    """
    if mode == "weekday":
        return weekday_residual(series, n_weeks=n_weeks)
    return series, None
