"""예시 입력 5종(data/sample/*.csv)을 만든다 - 개발자용.

data/integrated_dataset.csv 의 실제 일별 판매량을 '진짜 수요'로 보고,
빵집이 실제로 남기게 될 기록만 거꾸로 만들어 낸다.

- purchases.csv : 2년치 매입장부. 생산에 쓴 재료(+로스)를 다음 매입일까지 분량씩 산다.
- daily_log.csv : 최근 8주만 입력(앱을 쓰기 시작한 시점). 하루 한 값, 일부는 빈칸.
                  생산량보다 수요가 많으면 품절 시각을, 적으면 남은 개수를 적는다.
- events.csv    : 공휴일, 비, 크리스마스 시즌 + 예측 대상인 2026-01-01(신정).

    python make_sample_inputs.py            # data/sample/ 에 저장
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecast.config import Config

ROOT = Path(__file__).parent
SOURCE = ROOT / "data" / "integrated_dataset.csv"
OUT_DIR = ROOT / "data" / "sample"

LOG_DAYS = 56
LOSS = 0.05

PRODUCTS = pd.DataFrame([
    # 제품명, 판매가, 원가, 회수가(다음 날 할인가 등)
    ("크루아상", 3200, 1100, 0),
    ("식빵", 5000, 1500, 2500),
    ("단팥빵", 1600, 500, 0),
    ("소보로빵", 1500, 450, 0),
    ("소금빵", 2000, 600, 0),
], columns=["product", "price", "cost", "salvage"])

# 제품 1개당 재료량 (밀가루, 버터, 설탕, 팥앙금 kg, 계란 개)
RECIPE = {
    "크루아상": {"밀가루": 0.060, "버터": 0.035, "설탕": 0.006, "계란": 0.15},
    "식빵": {"밀가루": 0.300, "버터": 0.015, "설탕": 0.020},
    "단팥빵": {"밀가루": 0.050, "버터": 0.004, "설탕": 0.008, "계란": 0.10, "팥앙금": 0.060},
    "소보로빵": {"밀가루": 0.070, "버터": 0.012, "설탕": 0.020, "계란": 0.20},
    "소금빵": {"밀가루": 0.055, "버터": 0.015, "설탕": 0.003},
}

# 재료별 매입 요일(0=월). 설탕은 격주.
BUY_DAYS = {"밀가루": [0, 3], "버터": [1], "설탕": [0], "계란": [0, 2, 4], "팥앙금": [4]}
PACK = {"밀가루": 1.0, "버터": 0.5, "설탕": 1.0, "계란": 30, "팥앙금": 1.0}


def build_sample(seed: int = 7):
    rng = np.random.default_rng(seed)
    cfg = Config()
    raw = pd.read_csv(SOURCE, parse_dates=["date"])

    demand = (raw[raw["item"].isin(PRODUCTS["product"])]
              .groupby(["date", "item"])["sales_qty"].sum().unstack().sort_index())
    dates = demand.index
    log_start = dates[-LOG_DAYS]

    # 평소 생산량: 기록 시작 전 4주 평균 수요의 1.15배
    before = demand.loc[log_start - pd.Timedelta(days=28): log_start - pd.Timedelta(days=1)]
    usual = (before.mean() * 1.15).round().astype(int)
    products = PRODUCTS.assign(usual_qty=PRODUCTS["product"].map(usual))

    # ---- 생산량 (실제로 만든 개수) ----
    hist_prod = (demand.shift(1).rolling(14, min_periods=1).mean() * 1.1).round().bfill()
    production = hist_prod.copy()
    log_rows = []
    hours = list(range(cfg.open_hour, cfg.close_hour))
    prof = np.array([cfg.open_profile()[h] for h in hours])

    for d in dates[dates >= log_start]:
        for p in products["product"]:
            P = int(usual[p])
            logged_prod = ""
            if rng.random() < 0.12:                       # 가끔 평소와 다르게 만든 날
                P = int(round(P * rng.choice([0.8, 0.9, 1.15, 1.25])))
                logged_prod = str(P)
            production.loc[d, p] = P

            D = float(demand.loc[d, p])
            if D < P:
                result = str(int(P - D))
            else:
                # 그날의 시간대별 판매 흐름(기본 프로필 주변으로 흔들림)으로 품절 시각을 계산
                hourly = rng.dirichlet(prof * 60) * D
                cum = np.cumsum(hourly)
                k = int(np.searchsorted(cum, P))
                prev = cum[k - 1] if k > 0 else 0.0
                t = hours[k] + (P - prev) / hourly[k] if k < len(hours) else cfg.close_hour
                if t >= cfg.close_hour - 0.5:
                    result = "0"
                else:
                    mins = int(round((t % 1) * 60 / 10) * 10)
                    h = int(t) + mins // 60
                    result = f"{h}:{mins % 60:02d}"
            if rng.random() < 0.12:                       # 입력 안 한 날
                result, logged_prod = "", ""
            log_rows.append((d.strftime("%Y-%m-%d"), p, result, logged_prod))

    daily_log = pd.DataFrame(log_rows, columns=["date", "product", "result", "production"])

    # ---- 매입장부 ----
    recipe = pd.DataFrame(RECIPE).T.fillna(0.0)
    used = (production[recipe.index] @ recipe) * (1 + LOSS)          # 날짜 x 재료
    purchases = []
    for ing, wds in BUY_DAYS.items():
        buy = [d for d in dates if d.weekday() in wds]
        if ing == "설탕":
            buy = buy[::2]
        for a, b in zip(buy, buy[1:] + [dates[-1] + pd.Timedelta(days=1)]):
            q = used.loc[a: b - pd.Timedelta(days=1), ing].sum() * rng.normal(1.0, 0.03)
            q = max(round(q / PACK[ing]) * PACK[ing], PACK[ing])
            purchases.append((a.strftime("%Y-%m-%d"), ing, q))
    purchases = pd.DataFrame(purchases, columns=["date", "ingredient", "qty"]).sort_values("date")

    # ---- 이벤트 ----
    day = raw.groupby("date").agg(holiday=("is_holiday", "first"),
                                  precip=("precip_type", "first"),
                                  season=("season_period", "first"))
    ev = [(d, "holiday") for d in day.index[day["holiday"]]]
    ev += [(d, "rain") for d in day.index[day["precip"] == "비"]]
    ev += [(d, "event") for d in day.index[day["season"] == "크리스마스시즌"]]
    ev = pd.DataFrame(ev, columns=["date", "tag"])
    ev["date"] = ev["date"].dt.strftime("%Y-%m-%d")
    ev = pd.concat([ev, pd.DataFrame([("2026-01-01", "holiday")], columns=["date", "tag"])])
    ev = ev.sort_values(["date", "tag"]).reset_index(drop=True)

    recipe_long = (recipe.stack().rename("amount").reset_index()
                   .rename(columns={"level_0": "product", "level_1": "ingredient"}))
    recipe_long = recipe_long[recipe_long["amount"] > 0]

    files = {
        "products": products[["product", "price", "cost", "salvage", "usual_qty"]],
        "recipe": recipe_long,
        "purchases": purchases,
        "daily_log": daily_log,
        "events": ev,
    }
    return files, demand


def main():
    files, _ = build_sample()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in files.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name:10s} {len(df):5d}행 -> {OUT_DIR / (name + '.csv')}")


if __name__ == "__main__":
    main()
