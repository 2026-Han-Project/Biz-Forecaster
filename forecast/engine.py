"""매입 -> 재료 사용량 -> 역레시피 -> 진짜 수요 복원 -> 수준, 요일 학습 -> 시즌, 이벤트 -> 권장 생산량.

동작 원리 (단계별)
  1. 매입 -> 재료 일소비량   한 번 산 재료는 다음 매입일까지 고르게 쓴다고 본다.
  2. 역레시피 생산 추정  최근 bom_window_days 동안의 재료 소비량에서 제품별 생산량을 거꾸로 푼다.
                             '평소 생산량' 구성비가 결과를 붙잡아 준다(bom_lambda). -> 사전값
  3. 진짜 수요 복원          품절 시각과 시간대별 판매 비중으로 품절 뒤에 숨은 수요를 복원한다.
  4. 수준, 요일 계수 학습     요일 효과를 뺀 뒤 EWMA로 수준을 학습하고, 관측값과 사전값을
                             n/(n+k) 비율로 섞는다. 요일 계수도 같은 방식으로 매장 평균 쪽에 당긴다.
  5. 시즌, 이벤트 보정        매입 이력이 1년 이상이면 작년 같은 시기의 생산 흐름 변화를 곱하고,
                             공휴일, 비 같은 이벤트 배수를 곱한다.
  6. 권장 생산량             목표 서비스수준 = (판매가-원가)/(판매가-회수가) 분위수만큼 생산한다.

Streamlit에 의존하지 않으므로 UI 없이 단독으로 테스트할 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from forecast.config import Config, WEEKDAY_NAMES
from forecast.inputs import Inputs

OBSERVED_KINDS = ("남음", "품절", "마감 품절")


# =============================================================================
# 1. 매입 -> 재료 일소비량
# =============================================================================
def daily_usage(purchases: pd.DataFrame, until: pd.Timestamp | None = None) -> pd.DataFrame:
    """재료별 하루 사용량(달력일 기준). 추정할 수 없는 날은 NaN.

    i번째 매입량은 다음 매입 전날까지 고르게 쓴 것으로 본다.
    마지막 매입은 다음 매입일을 모르므로 그 재료의 평소 매입 간격(중앙값)만큼 쓴다고 본다.
    매입이 한 번뿐인 재료는 간격을 알 수 없어 추정하지 않는다.
    """
    p = purchases
    if until is not None:
        p = p[p["date"] <= until]
    if p.empty:
        return pd.DataFrame()
    p = p.groupby(["ingredient", "date"], as_index=False)["qty"].sum()

    series = {}
    for ing, g in p.groupby("ingredient"):
        dates = list(g["date"])
        qtys = list(g["qty"])
        if len(dates) < 2:
            continue
        gaps = [(b - a).days for a, b in zip(dates[:-1], dates[1:])]
        typical_gap = int(max(round(float(np.median(gaps))), 1))
        gaps.append(typical_gap)
        parts = []
        for d, q, gap in zip(dates, qtys, gaps):
            idx = pd.date_range(d, periods=gap, freq="D")
            parts.append(pd.Series(q / gap, index=idx))
        series[ing] = pd.concat(parts)
    if not series:
        return pd.DataFrame()
    out = pd.DataFrame(series)
    full = pd.date_range(out.index.min(), out.index.max(), freq="D")
    return out.reindex(full)


# =============================================================================
# 2. 역레시피 생산 추정
# =============================================================================
@dataclass
class BomEstimate:
    production: pd.Series              # 제품별 영업일 하루 평균 생산량
    method: str                        # '역레시피' | '평소 생산량'
    scale: float = 1.0                 # 평소 생산량 대비 전체 규모 (s)
    usage: pd.Series = field(default_factory=pd.Series)    # 재료별 영업일 하루 사용량(로스 제외)
    fitted: pd.Series = field(default_factory=pd.Series)   # 추정 생산량으로 되짚은 재료 사용량
    window: tuple = (None, None)


def reverse_recipe(usage: pd.DataFrame, recipe: pd.DataFrame, usual: pd.Series,
                   cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> BomEstimate:
    """[start, end] 기간 재료 소비량에서 제품별 하루 생산량을 역산한다.

    풀이: 비음수 최소제곱
        min_x  sum_j ((R x - u)_j / u_j)^2  +  lambda sum_p ((x_p - s, x0_p) / (s, x0_p))^2
    - R: 제품 1개당 재료량, u: 재료 사용량(로스 제외), x0: 평소 생산량
    - s: 평소 생산 '구성비'를 그대로 두고 규모만 맞춘 배율. lambda(bom_lambda)는 이 구성비를 믿는 정도다.
    - 재료마다 단위(kg, 개)가 달라 상대오차로 맞춘다.
    재료 4~5개만으로도 버터=크루아상, 팥앙금=단팥빵처럼 제품을 가리키는 재료가 있어 꽤 잘 구분된다.
    계산이 불안정할 때는(재료 수 < 제품 수 등) 두 번째 항이 '평소 생산량' 쪽으로 붙잡아 준다.
    """
    fallback = BomEstimate(usual.astype(float).copy(), "평소 생산량", window=(start, end))
    if usage.empty or recipe.empty:
        return fallback

    win = usage.loc[start:end]
    n_days = (end - start).days + 1
    if win.empty:
        return fallback
    coverage = win.notna().sum()
    ings = [j for j in recipe.columns
            if j in win.columns and coverage[j] >= max(7, n_days // 2)]
    u = win[ings].mean()
    ings = [j for j in ings if u[j] > 0]
    prods = [p for p in usual.index if p in recipe.index and recipe.loc[p, ings].sum() > 0]
    if not ings or not prods:
        return fallback

    # 재료는 달력일 내내 나눠 썼다고 봤으므로, 영업일 하루 기준으로 환산하고 로스를 뺀다.
    per_open_day = 7.0 / len(cfg.open_weekdays())
    target = u[ings] * per_open_day / (1.0 + cfg.loss_rate)

    A = recipe.loc[prods, ings].T.to_numpy(dtype=float)            # 재료 x 제품
    x0 = np.maximum(usual[prods].to_numpy(dtype=float), 1e-6)
    t = target.to_numpy(dtype=float)
    Ax0_rel = (A @ x0) / t
    scale = float(Ax0_rel.sum() / (Ax0_rel @ Ax0_rel)) if (Ax0_rel @ Ax0_rel) > 0 else 1.0

    lam = math.sqrt(max(cfg.bom_lambda, 0.0))
    M = np.vstack([A / t[:, None], np.diag(lam / (scale * x0))])
    b = np.concatenate([np.ones(len(ings)), np.full(len(prods), lam)])
    x, _ = nnls(M, b)

    production = usual.astype(float).copy()
    production.loc[prods] = x
    return BomEstimate(
        production=production, method="역레시피", scale=scale,
        usage=target, fitted=pd.Series(A @ x, index=ings), window=(start, end),
    )


# =============================================================================
# 3. 진짜 수요 복원
# =============================================================================
def parse_result(raw) -> tuple[str, float | None]:
    """daily_log 결과 칸 해석.

    '3' -> ('남음', 3) / '15:30' -> ('품절', 15.5) / '0' -> ('마감 품절', None)
    빈칸 -> ('미입력', None) / 그 밖 -> ('해석 불가', None)
    """
    s = "" if raw is None else str(raw).strip()
    if s == "" or s.lower() in ("nan", "none", "nat"):
        return "미입력", None
    if ":" in s:
        try:
            parts = [float(x) for x in s.split(" ")[-1].split(":")]
            hours = parts[0] + (parts[1] if len(parts) > 1 else 0) / 60 \
                + (parts[2] if len(parts) > 2 else 0) / 3600
            return "품절", hours
        except ValueError:
            return "해석 불가", None
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return "해석 불가", None
    if v < 0:
        return "해석 불가", None
    if v == 0:
        return "마감 품절", None
    return "남음", v


def cumulative_share(hour: float, cfg: Config) -> float:
    """영업 시작부터 hour 시각까지 하루 수요가 들어오는 비율."""
    if hour <= cfg.open_hour:
        return 0.0
    if hour >= cfg.close_hour:
        return 1.0
    prof = cfg.open_profile()
    h = int(math.floor(hour))
    return sum(v for k, v in prof.items() if k < h) + prof.get(h, 0.0) * (hour - h)


def restore_demand(daily_log: pd.DataFrame, products: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """마감 기록 한 줄을 '그날의 진짜 수요'로 바꾼다.

    - 남음 n개     : 수요 = 생산량 - n  (관측 그대로)
    - hh:mm 품절   : 수요 = 생산량 / (그 시각까지 판매 비중). 단 생산량 x max_uplift 까지만,
                     그리고 최소 생산량 x close_uplift (품절이면 수요는 생산량보다 컸다).
    - 0 (마감 품절): 수요 = 생산량 x close_uplift
    - 빈칸         : 입력 안 함 -> 건너뜀
    생산량 칸이 비어 있으면 평소 생산량을 쓴다.
    """
    cols = ["date", "product", "production", "result", "kind", "sold_out_at",
            "share_at_sellout", "demand", "censored"]
    if daily_log.empty:
        return pd.DataFrame(columns=cols).astype(
            {"date": "datetime64[ns]", "demand": float, "censored": bool})

    rows = []
    for r in daily_log.itertuples(index=False):
        kind, val = parse_result(r.result)
        if r.product not in products.index:
            kind = "미등록 제품"
        usual = float(products.loc[r.product, "usual_qty"]) if r.product in products.index else np.nan
        P = float(r.production) if pd.notna(r.production) else usual
        demand, censored, share, at = np.nan, False, np.nan, np.nan
        if kind == "남음":
            demand = max(P - val, 0.0)
        elif kind == "품절":
            at = val
            share = cumulative_share(val, cfg)
            restored = P / share if share > 0 else np.inf
            demand = float(np.clip(restored, P * cfg.close_uplift, P * cfg.max_uplift))
            censored = True
        elif kind == "마감 품절":
            demand = P * cfg.close_uplift
            censored = True
        rows.append([r.date, r.product, P, r.result, kind, at, share, demand, censored])

    out = pd.DataFrame(rows, columns=cols)
    # 같은 날 같은 제품을 두 번 적었으면 마지막 줄을 쓴다.
    return out.drop_duplicates(["date", "product"], keep="last").sort_values(["date", "product"]) \
              .reset_index(drop=True)


# =============================================================================
# 4~6. 학습 모델
# =============================================================================
def service_level(price: float, cost: float, salvage: float, cfg: Config) -> float:
    """목표 서비스수준(품절이 나지 않을 확률) = (판매가 - 원가) / (판매가 - 회수가)."""
    lo, hi = cfg.sl_bounds
    denom = price - salvage
    raw = (price - cost) / denom if denom > 0 else hi
    return float(np.clip(raw, lo, hi))


def _round_up(x: float) -> int:
    return int(math.ceil(x - 1e-9)) if x > 0 else 0


@dataclass
class ForecastModel:
    cfg: Config
    inputs: Inputs
    origin: pd.Timestamp
    usage: pd.DataFrame
    bom: BomEstimate
    demand: pd.DataFrame          # 복원된 마감 기록 (origin 이전)
    store_weekday: pd.Series      # 매장 전체 요일 계수
    weekday: pd.DataFrame         # 제품 x 요일 계수
    levels: pd.DataFrame          # 제품별 사전값, 관측 수준, 가중치, 학습 수준, CV
    season_enabled: bool
    notes: list = field(default_factory=list)
    _season_cache: dict = field(default_factory=dict, repr=False)

    # ---------------- 5. 시즌, 이벤트 ----------------
    def event_tags(self, date: pd.Timestamp) -> list[str]:
        ev = self.inputs.events
        if ev.empty:
            return []
        return sorted(set(ev.loc[ev["date"] == date, "tag"]))

    def event_factor(self, tags) -> float:
        f = 1.0
        for t in tags:
            f *= self.cfg.event_factors.get(t, 1.0)
        return f

    def season_factors(self, date: pd.Timestamp) -> pd.Series:
        """작년 '지금 -> 예측일' 사이 생산량이 얼마나 늘거나 줄었는지(매입 흐름 기준)."""
        products = self.inputs.products.index
        if not self.season_enabled:
            return pd.Series(1.0, index=products)
        if date in self._season_cache:
            return self._season_cache[date]
        half = pd.Timedelta(days=self.cfg.bom_window_days // 2)
        year = pd.Timedelta(days=364)       # 52주 - 요일을 맞춘다
        usual = self.inputs.products["usual_qty"]

        def around(center):
            return reverse_recipe(self.usage, self.inputs.recipe, usual, self.cfg,
                                  center - half, center + half - pd.Timedelta(days=1))

        now_ly, target_ly = around(self.origin - year), around(date - year)
        if now_ly.method != "역레시피" or target_ly.method != "역레시피":
            ratio = pd.Series(1.0, index=products)
        else:
            base = now_ly.production.replace(0, np.nan)
            ratio = (target_ly.production / base).fillna(1.0)
        lo, hi = self.cfg.season_bounds
        ratio = ratio.clip(lo, hi)
        self._season_cache[date] = ratio
        return ratio

    # ---------------- 예측 1건 ----------------
    def predict(self, product: str, date, tags: list[str] | None = None,
                weekday: int | None = None) -> dict:
        """제품 하나, 날짜 하나의 예측 수요와 권장 생산량.

        tags 를 주면 events.csv 대신 그 이벤트로, weekday 를 주면 그 요일로 계산한다(What-if).
        """
        date = pd.Timestamp(date).normalize()
        wd = date.weekday() if weekday is None else int(weekday)
        tags = self.event_tags(date) if tags is None else list(tags)
        prod = self.inputs.products.loc[product]
        lv = self.levels.loc[product]
        base = {
            "date": date, "product": product, "weekday": WEEKDAY_NAMES[wd], "tags": tags,
            "prior": lv["prior"], "level": lv["level"], "n_obs": int(lv["n_obs"]),
            "usual_qty": float(prod["usual_qty"]),
        }
        if wd in self.cfg.closed_weekdays:
            return {**base, "closed": True, "weekday_coef": 0.0, "season": 1.0,
                    "event": 1.0, "mu": 0.0, "cv": lv["cv"], "service_level": np.nan,
                    "production": 0}

        wcoef = float(self.weekday.loc[product, wd])
        season = float(self.season_factors(date).get(product, 1.0))
        event = self.event_factor(tags)
        mu = float(lv["level"]) * wcoef * season * event

        sl = service_level(prod["price"], prod["cost"], prod["salvage"], self.cfg)
        z = NormalDist().inv_cdf(sl)
        cv = float(lv["cv"])
        q = _round_up(mu * (1.0 + z * cv))
        return {**base, "closed": False, "weekday_coef": wcoef, "season": season,
                "event": event, "mu": mu, "cv": cv, "service_level": sl, "z": z,
                "production": q}

    def plan(self, dates) -> pd.DataFrame:
        rows = [self.predict(p, d) for d in dates for p in self.inputs.products.index]
        return pd.DataFrame(rows)

    def ingredient_needs(self, plan: pd.DataFrame) -> pd.DataFrame:
        """권장 생산량을 만들 때 필요한 재료량(로스 포함), 날짜 x 재료."""
        recipe = self.inputs.recipe
        if recipe.empty or plan.empty:
            return pd.DataFrame()
        q = plan.pivot_table(index="date", columns="product", values="production",
                             aggfunc="sum", fill_value=0)
        common = [p for p in q.columns if p in recipe.index]
        if not common:
            return pd.DataFrame()
        return (q[common] @ recipe.loc[common]) * (1.0 + self.cfg.loss_rate)


def _learn_weekday(obs: pd.DataFrame, products, cfg: Config):
    """제품 x 요일 계수. 관측이 적으면 매장 평균(-> 사장님 요일 감각) 쪽에 무게를 둔다.

    매장: c_w  = (N_w, 관측_w + k, weekday_prior_w) / (N_w + k)
    제품: c_pw = (n_pw, 관측_pw + k, c_w) / (n_pw + k)        (k = k_weekday)
    평균 영업일이 1이 되도록 정규화한다.
    """
    open_wd = cfg.open_weekdays()
    k = cfg.k_weekday

    def normalize(s: pd.Series) -> pd.Series:
        s = s.reindex(range(7)).fillna(0.0)
        m = s[open_wd].mean()
        s = s / m if m > 0 else pd.Series(1.0, index=range(7))
        s[cfg.closed_weekdays] = 0.0
        return s

    prior = normalize(pd.Series(cfg.weekday_prior, index=range(7)))
    store = prior.copy()
    coefs = pd.DataFrame({wd: prior[wd] for wd in range(7)}, index=list(products))
    if obs.empty:
        return store, coefs

    obs = obs.copy()
    obs["wd"] = obs["date"].dt.weekday
    for _ in range(3):      # 제품 평균 <-> 요일 계수를 번갈아 맞춘다(입력 누락으로 요일이 치우쳐도 견디게)
        c = coefs.stack()
        obs["c"] = [c[(p, w)] for p, w in zip(obs["product"], obs["wd"])]
        mean_p = (obs["adj"] / obs["c"]).groupby(obs["product"]).mean()
        obs["ratio"] = obs["adj"] / obs["product"].map(mean_p).replace(0, np.nan)
        valid = obs.dropna(subset=["ratio"])

        per_day = valid.groupby("date")["ratio"].mean()
        per_day_wd = per_day.index.weekday
        N = per_day.groupby(per_day_wd).size().reindex(range(7), fill_value=0)
        m = per_day.groupby(per_day_wd).mean().reindex(range(7)).fillna(0.0)
        store = normalize((N * m + k * prior) / (N + k))

        n_pw = valid.groupby(["product", "wd"]).size()
        m_pw = valid.groupby(["product", "wd"])["ratio"].mean()
        new = {}
        for p in products:
            row = {}
            for w in range(7):
                n = n_pw.get((p, w), 0)
                row[w] = (n * m_pw.get((p, w), 0.0) + k * store[w]) / (n + k)
            new[p] = normalize(pd.Series(row))
        coefs = pd.DataFrame(new).T
    return store, coefs


def fit(inputs: Inputs, cfg: Config | None = None, origin=None) -> ForecastModel:
    """origin(포함)까지 들어온 입력으로 모델을 학습한다. origin 이 없으면 입력의 마지막 날."""
    cfg = cfg or Config()
    notes = []
    products = inputs.products

    last_dates = [d.max() for d in (inputs.daily_log.get("date"), inputs.purchases.get("date"))
                  if d is not None and len(d)]
    if origin is None:
        origin = max(last_dates) if last_dates else pd.Timestamp.today().normalize()
    origin = pd.Timestamp(origin).normalize()

    # ---- 1. 매입 -> 재료 일소비량 ----
    usage = daily_usage(inputs.purchases, until=origin)

    # ---- 2. 역레시피 -> 사전값 ----
    start = origin - pd.Timedelta(days=cfg.bom_window_days - 1)
    bom = reverse_recipe(usage, inputs.recipe, products["usual_qty"], cfg, start, origin)
    if bom.method != "역레시피":
        notes.append("매입, 레시피 자료가 부족해 '평소 생산량'을 사전값으로 썼습니다.")
    no_recipe = [p for p in products.index if p not in inputs.recipe.index]
    if no_recipe and bom.method == "역레시피":
        notes.append(f"레시피가 없는 제품은 평소 생산량을 사전값으로 썼습니다: {', '.join(no_recipe)}")
    prior = bom.production * (1.0 - cfg.prior_waste)

    # ---- 3. 진짜 수요 복원 ----
    log = inputs.daily_log[inputs.daily_log["date"] <= origin]
    demand = restore_demand(log, products, cfg)
    bad = demand[demand["kind"].isin(["해석 불가", "미등록 제품"])]
    if len(bad):
        notes.append(f"읽지 못한 마감 기록 {len(bad)}줄은 건너뛰었습니다.")

    # ---- 4. 수준, 요일 계수 학습 ----
    obs = demand[demand["kind"].isin(OBSERVED_KINDS)].copy()
    obs = obs[~obs["date"].dt.weekday.isin(cfg.closed_weekdays)]
    ev = inputs.events
    if len(obs) and not ev.empty:
        f = ev.assign(f=ev["tag"].map(cfg.event_factors).fillna(1.0)).groupby("date")["f"].prod()
        obs["event_factor"] = obs["date"].map(f).fillna(1.0)
    else:
        obs["event_factor"] = 1.0
    obs["adj"] = obs["demand"] / obs["event_factor"]       # 이벤트 날의 튐이 수준을 끌고 가지 않게

    store_wd, weekday = _learn_weekday(obs, products.index, cfg)

    rows = {}
    for p in products.index:
        g = obs[obs["product"] == p].sort_values("date")
        y = (g["adj"] / [weekday.loc[p, d.weekday()] for d in g["date"]]).to_numpy()
        L, errs = None, []
        for v in y:
            if L is None:
                L = v
                continue
            if L > 0:
                errs.append(v / L - 1.0)
            L = cfg.alpha * v + (1 - cfg.alpha) * L
        n = len(y)
        w = n / (n + cfg.k_cred) if (n + cfg.k_cred) > 0 else 0.0
        level = w * L + (1 - w) * prior[p] if L is not None else prior[p]
        if len(errs) >= 3:
            ne = len(errs)
            cv_obs = float(np.std(errs, ddof=1))
            cv = (ne * cv_obs + cfg.k_cred * cfg.default_cv) / (ne + cfg.k_cred)
        else:
            cv_obs, cv = np.nan, cfg.default_cv
        rows[p] = {
            "prior_production": bom.production[p], "prior": prior[p],
            "ewma": L if L is not None else np.nan, "n_obs": n, "weight": w,
            "level": level, "cv_obs": cv_obs, "cv": float(np.clip(cv, 0.05, 1.0)),
            "n_censored": int(g["censored"].sum()),
        }
    levels = pd.DataFrame(rows).T.astype(float)
    levels[["n_obs", "n_censored"]] = levels[["n_obs", "n_censored"]].astype(int)

    # ---- 5. 시즌 보정 가능 여부 ----
    year_back = origin - pd.Timedelta(days=364 + cfg.bom_window_days)
    season_enabled = (not usage.empty and usage.index.min() <= year_back
                      and bom.method == "역레시피")

    unknown = sorted(set(ev["tag"]) - set(cfg.event_factors)) if not ev.empty else []
    if unknown:
        notes.append(f"배수가 정해지지 않은 이벤트 태그는 무시했습니다: {', '.join(unknown)}")

    return ForecastModel(cfg, inputs, origin, usage, bom, demand, store_wd, weekday,
                         levels, season_enabled, notes)


def next_open_days(model: ForecastModel, n: int = 1, start=None) -> list[pd.Timestamp]:
    """start(기본: 학습 기준일 다음 날)부터 휴무일을 건너뛴 영업일 n개."""
    d = pd.Timestamp(start).normalize() if start is not None else model.origin + pd.Timedelta(days=1)
    out = []
    while len(out) < n:
        if d.weekday() not in model.cfg.closed_weekdays:
            out.append(d)
        d += pd.Timedelta(days=1)
    return out
