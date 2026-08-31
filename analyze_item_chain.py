"""H2′ 검증 — 품목간 수요연쇄 (선행 품목 → 후행 품목 리드-래그 분석).

앱 화면에서는 뺐다. 소상공인이 매일 볼 화면이 아니라 "이 방법이 예측을 개선하는가"를
증명하는 연구용이기 때문이다. 발표·보고서에 필요할 때 이 스크립트로 재현한다.

  가설
    어떤 품목이 팔리면 며칠 뒤 다른 품목이 팔린다. 그 시차 관계를 예측에 넣으면
    각 품목을 따로 예측할 때보다 정확해진다.

  검증 3단계
    1. 시차 찾기   — 1~14일 중 두 계열의 교차상관이 가장 큰 간격 L*
    2. 인과성 검정 — Granger 검정. '같이 움직이는 것'과 '앞선 것이 뒤를 예측하는 것'을 가른다
    3. 예측 반영   — 선행 품목의 L*일 전 값을 후행 품목 예측에 더하고 WAPE가 줄었는지 본다

  사용법
    .venv\\Scripts\\python.exe analyze_item_chain.py
    .venv\\Scripts\\python.exe analyze_item_chain.py --lead 아메리카노 --lag 조각케이크
    .venv\\Scripts\\python.exe analyze_item_chain.py --by category
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "integrated_dataset.csv"
TEST_DAYS = 30          # 마지막 30일을 떼어 백테스트


def find_optimal_lag(leading, lagging, min_lag=1, max_lag=14):
    """leading(선행 일별 판매량) ↔ lagging(후행) 교차상관으로 최적 시차 L*를 탐색."""
    best_lag, best_corr = min_lag, 0.0
    corr_by_lag = {}
    for lag in range(min_lag, max_lag + 1):
        shifted = leading.shift(lag)
        valid = shifted.notna() & lagging.notna()
        if valid.sum() < 10:
            continue
        corr = np.corrcoef(shifted[valid], lagging[valid])[0, 1]
        corr_by_lag[lag] = corr
        if abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, corr
    return best_lag, best_corr, corr_by_lag


def granger_pvalue(lagging, leading, lag):
    """leading이 lagging을 Granger 인과하는지 검정, p-value 반환 (실패 시 None).

    상관은 '같이 움직인다'만 말해준다. Granger 검정은 '앞선 것이 뒤를 예측하는가'를 본다.
    p가 작을수록 선행 관계가 있다는 증거가 강하다.
    """
    data = pd.concat([lagging, leading], axis=1).dropna()
    data.columns = ['y', 'x']
    try:
        # statsmodels 0.15 에서 verbose 인자가 제거됐다. 예전 버전은 이 인자가 없으면
        # 검정 결과를 콘솔에 잔뜩 출력하므로, 버전을 보고 갈라서 호출한다.
        # (원래 코드는 verbose=False 를 그냥 넘겨 TypeError 가 났고,
        #  그걸 except 가 삼켜서 p-value 가 조용히 None 으로만 나왔다.)
        try:
            result = grangercausalitytests(data, maxlag=[lag])
        except TypeError:
            result = grangercausalitytests(data, maxlag=[lag], verbose=False)
        return result[lag][0]['ssr_ftest'][1]
    except Exception as e:
        print(f"  [경고] Granger 검정 실패: {type(e).__name__} — {e}")
        return None


def fit_item_chain_gain(lagging_train, leading_lagged_train):
    """후행 품목의 '자체 기저치 대비 잔차'를 선행 품목의 시차 신호로 회귀해 γ를 추정.

    'ŷ_final = ŷ_base x (1+Σβₖeₖ) + γ·x_item(t−L*)' 융합식의 가산항에 대응한다.
    """
    baseline = lagging_train.rolling(14, center=True, min_periods=7).median()
    baseline = baseline.bfill().ffill()
    residual = (lagging_train - baseline).fillna(0.0)

    reg = LinearRegression()
    X = leading_lagged_train.values.reshape(-1, 1)
    reg.fit(X, residual.values)
    return reg


def predict_item_chain(base_forecast, reg, leading_lagged_test):
    X = leading_lagged_test.values.reshape(-1, 1)
    residual_hat = reg.predict(X)
    return np.clip(np.asarray(base_forecast) + residual_hat, 0, None)


def compute_wape(actual, pred):
    """WAPE — 실제값 합 대비 절대오차 합의 비율(%). 작을수록 정확하다."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    denom = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / denom * 100) if denom else float("nan")


def main():
    ap = argparse.ArgumentParser(description="H2′ 품목간 수요연쇄 검증")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--by", default="item", choices=["item", "category"],
                    help="품목 단위로 볼지 카테고리 단위로 볼지")
    ap.add_argument("--lead", default=None, help="선행 품목/카테고리 (생략하면 전체 조합 탐색)")
    ap.add_argument("--lag", default=None, help="후행 품목/카테고리")
    ap.add_argument("--top", type=int, default=10, help="탐색 모드에서 보여줄 상위 조합 수")
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        raise SystemExit(f"파일을 찾을 수 없습니다: {csv}")

    df = pd.read_csv(csv, parse_dates=["date"])
    key = args.by
    if key not in df.columns:
        raise SystemExit(f"'{key}' 컬럼이 없습니다. 사용 가능: {list(df.columns)}")

    total_idx = df.groupby("date")["sales_qty"].sum().asfreq("D").index
    series = {
        name: g.groupby("date")["sales_qty"].sum().reindex(total_idx, fill_value=0.0)
        for name, g in df.groupby(key)
    }
    names = sorted(series)

    print()
    print(f"대상 단위 : {key}   ({len(names)}개)")
    print(f"기간      : {total_idx.min().date()} ~ {total_idx.max().date()} ({len(total_idx)}일)")

    if args.lead and args.lag:
        for n in (args.lead, args.lag):
            if n not in series:
                raise SystemExit(f"'{n}' 을(를) 찾을 수 없습니다. 사용 가능: {', '.join(names)}")
        pairs = [(args.lead, args.lag)]
    else:
        pairs = [(a, b) for a in names for b in names if a != b]

    rows = []
    for lead, lag_name in pairs:
        L, corr, _ = find_optimal_lag(series[lead], series[lag_name])
        if abs(corr) < 0.1:
            continue
        p = granger_pvalue(series[lag_name], series[lead], L)

        y = series[lag_name]
        x_lagged = series[lead].shift(L)
        train_y, test_y = y.iloc[:-TEST_DAYS], y.iloc[-TEST_DAYS:]
        train_x = x_lagged.iloc[:-TEST_DAYS].fillna(0.0)
        test_x = x_lagged.iloc[-TEST_DAYS:].fillna(0.0)

        # 기저 예측: 학습 구간 마지막 28일 평균을 유지 (가장 단순한 기준선)
        base = float(train_y.tail(28).mean())
        base_fc = np.full(len(test_y), base)

        reg = fit_item_chain_gain(train_y, train_x)
        chain_fc = predict_item_chain(base_fc, reg, test_x)

        w_base = compute_wape(test_y.values, base_fc)
        w_chain = compute_wape(test_y.values, chain_fc)
        rows.append({
            "선행": lead, "후행": lag_name, "시차(일)": L, "상관": round(corr, 3),
            "Granger p": None if p is None else round(p, 4),
            "기저 WAPE": round(w_base, 1), "연쇄 WAPE": round(w_chain, 1),
            "개선": round(w_base - w_chain, 1),
        })

    if not rows:
        print()
        print("상관이 0.1을 넘는 조합이 없습니다.")
        return

    res = pd.DataFrame(rows).sort_values("개선", ascending=False)
    pd.set_option("display.width", 200)
    print()
    print("=" * 88)
    print("  H2′ 검증 결과 — '개선'이 양수면 연쇄를 반영해 예측이 좋아진 것")
    print("=" * 88)
    print(res.head(args.top).to_string(index=False))

    n_better = int((res["개선"] > 0).sum())
    good = res[(res["개선"] > 0) & res["Granger p"].notna() & (res["Granger p"] < 0.05)]
    print()
    print(f"  전체 {len(res)}개 조합 중 예측이 개선된 조합 {n_better}개")
    print(f"  그중 Granger 인과성까지 통과(p<0.05)한 조합 {len(good)}개")
    print()
    if len(good):
        b = good.iloc[0]
        print(f"  가장 강한 관계: '{b['선행']}' -> '{b['후행']}' "
              f"(시차 {b['시차(일)']}일, WAPE {b['기저 WAPE']}% -> {b['연쇄 WAPE']}%)")
    else:
        print("  통계적으로 확실한 연쇄 관계는 확인되지 않았습니다.")
        print("  이것도 유효한 검증 결과입니다 — 가설이 기각된 것이지 분석이 실패한 게 아닙니다.")
    print()


if __name__ == "__main__":
    main()
