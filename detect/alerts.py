"""탐지 결과를 소상공인이 바로 읽고 행동할 수 있는 알림으로 옮긴다.

control_charts.py 는 '통계적으로 이상한가'를 판단하고, 이 파일은 그 결과를
'그래서 재고를 어떻게 하라는 것인가'로 번역한다. 두 관심사를 파일로 갈라놓은 이유는
화면에 3시그마·EWMA·CUSUM 같은 말이 한 글자도 나가지 않게 하기 위해서다.

번역 규칙
    많이 팔린 날  -> 재고가 부족했을 수 있다  -> 품절 위험, 생산·발주 늘리기 검토
    적게 팔린 날  -> 재고가 남았을 수 있다    -> 폐기 위험, 생산·발주 줄이기 검토
    전체 매출도 같이 흔들림 -> 날씨·휴일 같은 바깥 요인일 가능성
    이 품목만 흔들림        -> 진열·품질·경쟁 제품 등 그 품목의 문제일 가능성

'평소'의 기준은 직전 4주간 같은 요일의 평균이다. 전체 평균이 아니라 같은 요일을
쓰는 이유는, 빵집 판매량이 요일마다 크게 다르기 때문이다. 토요일 판매량을
평일 섞인 평균과 비교하면 멀쩡한 토요일이 매번 '이상'으로 잡힌다.
"""

import numpy as np
import pandas as pd

from detect.control_charts import run_all_detectors, weekday_residual

# 벗어난 정도에 따른 등급. 화면에서 색과 정렬 순서를 정한다.
SEVERITY_ORDER = {"높음": 0, "보통": 1, "낮음": 2}


def _josa(word, with_batchim, without_batchim):
    """받침 유무에 따라 조사를 고른다 ('단팥빵이' / '카스테라가').

    한글 음절은 유니코드에서 (초성, 중성, 종성) 순으로 배열돼 있어서,
    코드값에서 0xAC00을 뺀 뒤 28로 나눈 나머지가 0이면 받침이 없다.
    """
    if not word:
        return without_batchim
    ch = word[-1]
    if not ("가" <= ch <= "힣"):
        return without_batchim          # 영문·숫자로 끝나면 받침 없는 쪽으로
    return with_batchim if (ord(ch) - 0xAC00) % 28 else without_batchim


def build_context(df):
    """날짜별 '그 날 무슨 날이었나'를 뽑아둔다 (시즌·공휴일·날씨).

    이게 없으면 크리스마스 시즌 판매 급증이 그냥 '이상'으로만 뜬다. 소상공인 입장에서는
    예측 가능한 대목을 이상이라고 알려주는 셈이라 소음이 된다. 반대로 시즌을 붙여주면
    "작년 크리스마스엔 이만큼 팔렸으니 올해는 이만큼 준비하자"는 쓸모 있는 정보가 된다.

    필요한 컬럼이 없으면 빈 dict 를 돌려주고, 알림은 시즌 설명 없이 그대로 나간다.
    """
    if "date" not in df.columns:
        return {}

    cols = [c for c in ("season_period", "is_holiday", "precip_type") if c in df.columns]
    if not cols:
        return {}

    daily = df.groupby("date")[cols].first()
    ctx = {}
    for d, row in daily.iterrows():
        tags = []
        season = row.get("season_period")
        if isinstance(season, str) and season not in ("", "평시"):
            tags.append(season)
        if bool(row.get("is_holiday", False)):
            tags.append("공휴일")
        precip = row.get("precip_type")
        if isinstance(precip, str) and precip in ("비", "눈"):
            tags.append(precip + "오는 날")
        if tags:
            ctx[pd.Timestamp(d)] = tags
    return ctx


def _severity(pct, n_methods):
    """벗어난 비율과 몇 가지 방법이 함께 잡았는지로 등급을 매긴다."""
    a = abs(pct)
    if a >= 0.50 or n_methods >= 3:
        return "높음"
    if a >= 0.30 or n_methods >= 2:
        return "보통"
    return "낮음"


def _describe(item, date, actual, expected, pct, scope, tags=None):
    """알림 한 건을 사람이 읽는 문장으로 만든다.

    tags 는 그 날의 상황(크리스마스시즌·공휴일·비 등). 있으면 원인 설명에 붙인다.
    """
    day_kr = "월화수목금토일"[date.dayofweek]
    direction = "많이" if pct > 0 else "적게"
    label = item if item != "(전체 매출)" else "전체 매출"

    headline = (
        f"{date.month}월 {date.day}일({day_kr}) {label}{_josa(label, '이', '가')} "
        f"평소보다 {abs(pct) * 100:.0f}% {direction} 나갔습니다"
    )
    detail = (
        f"평소 이 요일 판매량은 {expected:,.0f}개 정도인데 "
        f"이 날은 {actual:,.0f}개였습니다."
    )

    if pct > 0:
        meaning = "재고가 부족했을 수 있습니다."
        action = "이런 날이 반복되면 생산·발주량을 늘리는 것을 검토하세요."
        icon = "📈"
    else:
        meaning = "재고가 남았을 수 있습니다."
        action = "이런 날이 반복되면 생산·발주량을 줄여 폐기를 줄이세요."
        icon = "📉"

    if tags:
        # 이유를 아는 날이면 그 이유를 먼저 말해준다.
        # 해마다 돌아오는 대목은 '문제'가 아니라 '준비할 일'이다.
        what = " · ".join(tags)
        cause = f"이 날은 **{what}**이었습니다. "
        if pct > 0:
            cause += "해마다 돌아오는 시기라면 내년 같은 때 미리 넉넉히 준비하세요."
            action = f"{what}에 맞춰 생산 계획을 미리 잡아두면 품절을 막을 수 있습니다."
        else:
            cause += "이런 날은 손님이 줄 수 있으니 그만큼만 준비하면 폐기를 줄일 수 있습니다."
            action = f"{what}에는 생산량을 미리 줄여두세요."
    elif scope == "전체 동반":
        cause = "이 날은 매장 전체 매출도 함께 흔들렸습니다. 날씨나 휴일 같은 바깥 요인일 수 있습니다."
    elif scope == "품목 단독":
        cause = "이 날 매장 전체 매출은 평소와 비슷했습니다. 이 품목만의 문제일 수 있습니다 (진열 위치, 품질, 경쟁 제품)."
    else:
        cause = ""

    return {"headline": headline, "detail": detail, "meaning": meaning,
            "action": action, "cause": cause, "icon": icon}


def build_alerts(daily_by_item, daily_total, target_items, recent_days=30,
                 min_expected=3.0, max_alerts=40, context=None):
    """품목별로 탐지를 돌려 알림 목록을 만든다.

    daily_by_item : {품목명: 일별 판매량 Series}
    daily_total   : 매장 전체 일별 매출 Series (바깥 요인 판단용)
    recent_days   : 최근 며칠 안의 이상만 보여줄지
    min_expected  : 평소 판매량이 이보다 적은 품목은 건너뛴다.
                    하루 1~2개 팔리는 품목은 0개인 날이 흔해 비율 변화가 무의미하다.

    반환: 심각도 순으로 정렬된 알림 dict 리스트
    """
    if len(daily_total) < 40:
        return []

    # 매장 전체가 흔들린 날 — 개별 품목의 원인을 가르는 데 쓴다.
    _, total_combined = run_all_detectors(daily_total, residual_mode="weekday")
    total_bad = set(total_combined.index[total_combined["이상여부"]])

    cutoff = daily_total.index.max() - pd.Timedelta(days=recent_days)
    alerts = []

    for item in target_items:
        series = daily_by_item.get(item)
        if series is None or len(series) < 40:
            continue
        if float(series.tail(90).mean()) < min_expected:
            continue

        _, combined = run_all_detectors(series, residual_mode="weekday")
        residual, expected = weekday_residual(series)

        hits = combined.index[combined["이상여부"] & (combined.index > cutoff)]
        for d in hits:
            exp = expected.get(d, np.nan)
            if pd.isna(exp) or exp < min_expected:
                continue
            actual = float(series.loc[d])
            pct = (actual - float(exp)) / float(exp)
            if abs(pct) < 0.20:
                # 20% 미만 차이는 통계적으로는 잡혀도 현장에서 의미가 없다.
                continue

            n_methods = int(combined.loc[d, "탐지방법수"])
            scope = "전체 동반" if d in total_bad else "품목 단독"
            tags = (context or {}).get(d)
            text = _describe(item, d, actual, float(exp), pct, scope, tags)

            severity = _severity(pct, n_methods)
            if tags:
                # 이유를 아는 날은 한 단계 낮춘다. 크리스마스 대목을 '긴급'으로 띄우면
                # 정작 원인 모를 급감이 그 아래 묻힌다.
                severity = {"높음": "보통", "보통": "낮음", "낮음": "낮음"}[severity]

            alerts.append({
                "date": d, "item": item, "actual": actual, "expected": float(exp),
                "pct": pct, "scope": scope, "n_methods": n_methods,
                "severity": severity, "tags": tags or [], "is_event": bool(tags), **text,
            })

    # 이유를 모르는 이상을 먼저 보여준다 — 그쪽이 확인이 필요한 쪽이다.
    alerts.sort(key=lambda a: (a["is_event"], SEVERITY_ORDER[a["severity"]], -abs(a["pct"])))
    return alerts[:max_alerts]


def trend_summary(series, weeks=4, min_expected=3.0):
    """최근 몇 주간 꾸준히 늘거나 줄고 있는지 본다.

    하루짜리 이상보다 재고 판단에 더 중요한 신호다. 하루 튄 건 우연일 수 있지만,
    4주 연속 줄고 있다면 발주량 자체를 손봐야 한다는 뜻이다.
    """
    if len(series) < weeks * 14:
        return None

    weekly = series.resample("W").sum()
    recent = weekly.tail(weeks)
    prev = weekly.tail(weeks * 2).head(weeks)
    if len(recent) < weeks or len(prev) < weeks:
        return None
    if float(prev.mean()) < min_expected * 7:
        return None

    change = (float(recent.mean()) - float(prev.mean())) / float(prev.mean())
    if abs(change) < 0.15:
        return None

    if change > 0:
        return {
            "change": change, "direction": "상승",
            "text": f"최근 {weeks}주 판매량이 그 이전 {weeks}주보다 {change * 100:.0f}% 늘었습니다.",
            "action": "발주·생산량을 늘리지 않으면 품절이 잦아질 수 있습니다.",
        }
    return {
        "change": change, "direction": "하락",
        "text": f"최근 {weeks}주 판매량이 그 이전 {weeks}주보다 {abs(change) * 100:.0f}% 줄었습니다.",
        "action": "지금 발주량을 유지하면 재고가 쌓이고 폐기가 늘 수 있습니다.",
    }


def weekday_guide(series, weeks=4):
    """요일별 평균 판매량 — '다음 주 화요일엔 몇 개 준비할까'에 바로 답하는 표."""
    recent = series.tail(weeks * 7)
    if len(recent) < 14:
        return None
    names = ["월", "화", "수", "목", "금", "토", "일"]
    g = recent.groupby(recent.index.dayofweek).agg(["mean", "max"])
    rows = []
    for wd in range(7):
        if wd not in g.index:
            continue
        rows.append({
            "요일": names[wd],
            "평균 판매량": round(float(g.loc[wd, "mean"]), 1),
            "가장 많이 팔린 날": int(g.loc[wd, "max"]),
            "권장 준비량": int(np.ceil(g.loc[wd, "mean"] * 1.1)),
        })
    return pd.DataFrame(rows)
