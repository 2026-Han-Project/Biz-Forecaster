import streamlit as st
import pandas as pd
import re
import numpy as np
from datetime import timedelta
from typing import TypedDict, List
import io
import json
import sqlite3
from pathlib import Path
from agent.brief_llm import generate_briefing
from agent.mock_shap import get_contributions, format_contributions
from agent.whatif_llm import parse_events_llm, stream_reply

from detect.control_charts import (
    shewhart_3sigma,
    ewma_chart,
    cusum_chart,
    isolation_forest_flags,
    run_all_detectors,
    residualize,
    weekday_residual,
)


from detect.alerts import build_alerts, build_context, trend_summary, weekday_guide

# --- 계정 인증 (소상공인별 로그인) ---
from auth.ui import (
    require_login,
    render_sidebar_account,
    render_account_page,
    ACCOUNT_VIEW_KEY,
)
from auth import store as auth_store
import plotly.graph_objects as go
import plotly.express as px

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests

try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression
    import xgboost as xgb

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import logging as _logging
    from prophet import Prophet

    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)
    _logging.getLogger("prophet").setLevel(_logging.WARNING)

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    _logging.getLogger("pytorch_lightning").setLevel(_logging.WARNING)
    _logging.getLogger("lightning").setLevel(_logging.WARNING)

    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# 1. 페이지 설정
st.set_page_config(page_title="Biz-Forecaster", page_icon="📈", layout="wide")
CSS_PATH = Path(__file__).parent / "main.css"


def load_css(path=CSS_PATH):
    """main.css 를 읽어 페이지에 주입한다. 파일이 없어도 앱은 기본 스타일로 동작한다."""
    if path.exists():
        st.markdown(
            f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True
        )


load_css()


def bf_html(markup):
    """들여쓴 HTML 이 마크다운 코드블록으로 보이지 않도록 줄 앞 공백을 지우고 렌더링한다."""
    st.markdown(
        "".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True
    )


import plotly.io as pio

BF_BLUE, BF_NAVY, BF_SKY, BF_GRAY, BF_RED = (
    "#3B82F6",
    "#1E3A8A",
    "#93C5FD",
    "#94A3B8",
    "#EF4444",
)
BF_PALETTE = [
    "#3B82F6",
    "#0EA5E9",
    "#1E3A8A",
    "#6366F1",
    "#14B8A6",
    "#60A5FA",
    "#F59E0B",
    "#94A3B8",
    "#8B5CF6",
    "#22C55E",
]
pio.templates["bf"] = go.layout.Template(
    layout=dict(
        font=dict(
            family="Pretendard Variable, Pretendard, Malgun Gothic, sans-serif",
            color="#0F172A",
        ),
        colorway=BF_PALETTE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="#EEF2F7", linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#EEF2F7", linecolor="#E2E8F0"),
        title=dict(font=dict(size=16, color="#1E293B")),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#3B82F6"),
    )
)
pio.templates.default = "plotly_white+bf"
px.defaults.color_discrete_sequence = BF_PALETTE
px.defaults.color_continuous_scale = "Blues"

# 사이드바
with st.sidebar:
    bf_html("""
        <div class="bf-brand">
          <div class="bf-brand-logo">BF</div>
          <div>
            <div class="bf-brand-name">Biz-Forecaster</div>
            <div class="bf-brand-sub">매장 수요 예측</div>
          </div>
        </div>
        """)

# 1-1. 로그인 게이트
SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "integrated_dataset.csv"

CURRENT_USER = require_login()
if CURRENT_USER is None:
    st.stop()
if st.session_state.get(ACCOUNT_VIEW_KEY):
    render_sidebar_account()
    render_account_page()
    st.stop()


@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    daily = (
        df.groupby(["date", "item"], as_index=False)["sales_qty"]
        .sum()
        .rename(columns={"date": "Date", "item": "Item", "sales_qty": "Quantity"})
    )
    daily["Quantity"] = daily["Quantity"].astype(float)
    return daily[["Item", "Date", "Quantity"]]


def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()


# 2. 품목 등급 계산 함수
def calculate_item_tiers(df):
    # 품목별 총 판매량 집계
    item_stats = df.groupby("Item")["Quantity"].sum().reset_index()
    item_stats = item_stats.sort_values("Quantity", ascending=False)

    # 순위 백분율 계산
    item_stats["Rank_Pct"] = item_stats["Quantity"].rank(pct=True, ascending=False)

    # 등급 부여
    def get_tier(pct):
        if pct <= 0.1:
            return "시그니처 (상위 10%)"
        elif pct <= 0.3:
            return "인기 (상위 30%)"
        elif pct <= 0.6:
            return "스테디셀러 (상위 60%)"
        else:
            return "일반"

    item_stats["Tier"] = item_stats["Rank_Pct"].apply(get_tier)
    return item_stats[["Item", "Tier", "Quantity"]]


# 3. 예측 모델 함수 (추세 반영 강화)
def predict_linear_trend_force(series, weeks=5):
    try:
        lookback = 8
        recent_data = series[-lookback:] if len(series) >= lookback else series
        n = len(recent_data)
        if n < 2:
            return [series.mean()] * weeks

        x = np.arange(n)
        y = recent_data
        slope, intercept = np.polyfit(x, y, 1)

        future_x = np.arange(n, n + weeks)
        forecast = slope * future_x + intercept
        return forecast
    except:
        return [series.mean()] * weeks


def predict_holt_trend(series, weeks=5):
    try:
        if len(series) >= 4:
            model = ExponentialSmoothing(
                series, trend="add", seasonal=None, damped_trend=False
            ).fit(optimized=True)
            return np.asarray(model.forecast(weeks))
        else:
            return predict_linear_trend_force(series, weeks)
    except Exception:
        return predict_linear_trend_force(series, weeks)


def predict_arima_trend(series, weeks=5):
    try:
        model = ARIMA(series, order=(1, 1, 1)).fit()
        return np.asarray(model.forecast(steps=weeks))
    except Exception:
        return predict_linear_trend_force(series, weeks)


def create_ml_dataset(series, window_size=4):
    X, y = [], []
    s_list = list(series)
    for i in range(len(s_list) - window_size):
        X.append(s_list[i : i + window_size])
        y.append(s_list[i + window_size])
    return np.array(X), np.array(y)


def predict_ml_recursive(model, last_window, weeks=5):
    preds = []
    curr = list(last_window)
    for _ in range(weeks):
        in_row = np.array(curr[-len(last_window) :]).reshape(1, -1)
        pred = model.predict(in_row)[0]
        preds.append(pred)
        curr.append(pred)
    return preds


def predict_rf(series, weeks=5):
    if not ML_AVAILABLE:
        return [0] * weeks
    win = 4
    if len(series) < win + 2:
        return predict_linear_trend_force(series, weeks)
    X, y = create_ml_dataset(series, win)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return predict_ml_recursive(model, series[-win:], weeks)


def predict_xgboost(series, weeks=5):
    if not ML_AVAILABLE:
        return [0] * weeks
    win = 4
    if len(series) < win + 2:
        return predict_linear_trend_force(series, weeks)
    X, y = create_ml_dataset(series, win)
    model = xgb.XGBRegressor(
        n_estimators=100, objective="reg:squarederror", random_state=42
    )
    model.fit(X, y)
    return predict_ml_recursive(model, series[-win:], weeks)


def predict_prophet(dated_series, periods, freq="D"):
    """Layer 1 기저모델(Prophet). dated_series는 DatetimeIndex를 가진 pd.Series여야
    실제 달력(연간·주간 계절성)을 학습할 수 있다. 실패·미설치·데이터 부족 시
    predict_linear_trend_force로 폴백한다."""
    values = np.asarray(
        dated_series.values if hasattr(dated_series, "values") else dated_series,
        dtype=float,
    )
    if not PROPHET_AVAILABLE or len(values) < 10 or not hasattr(dated_series, "index"):
        return predict_linear_trend_force(values, periods)
    try:
        train_df = pd.DataFrame({"ds": dated_series.index, "y": values})
        span_periods = len(train_df)
        is_weekly = freq.startswith("W")
        model = Prophet(
            weekly_seasonality=(not is_weekly),
            yearly_seasonality=(span_periods >= (104 if is_weekly else 365)),
            daily_seasonality=False,
        )
        model.fit(train_df)
        future = model.make_future_dataframe(
            periods=periods, freq=freq, include_history=False
        )
        forecast = model.predict(future)
        return forecast["yhat"].to_numpy()[:periods]
    except Exception:
        return predict_linear_trend_force(values, periods)


def predict_prophet_with_events(train_series, train_events, test_events, freq="D"):
    """Layer 2(H1) — 이벤트를 사후 곱셈 보정이 아니라 Prophet 회귀변수로
    기저모델 안에 직접 투입한다.

    기존 곱셈형 게이팅 `ŷ_base×(1+Σβₖeₖ)`은 Prophet이 주간 계절성으로 이미
    학습한 주말 효과를 다시 곱해 이벤트 효과를 **이중 계상**한다(실측상
    β_weekend가 품목마다 +0.35~+0.43로 추정되어, 주말마다 예측치를 35~43%
    추가 증폭시키고 있었다). `add_regressor()`로 넣으면 추세·계절성·이벤트가
    한 번의 적합에서 동시에 분리 추정되므로 이 중복이 구조적으로 사라진다.

    실측(일별, 검증 14일, 4품목 평균 WAPE): 기저 31.7% / 곱셈형 게이팅 35.8%
    / 이벤트 회귀 22.5%.

    미설치·데이터 부족·학습 실패 시 predict_prophet(이벤트 미반영)으로 폴백한다.
    """
    periods = len(test_events)
    values = np.asarray(
        train_series.values if hasattr(train_series, "values") else train_series,
        dtype=float,
    )
    if not PROPHET_AVAILABLE or len(values) < 10 or not hasattr(train_series, "index"):
        return predict_prophet(train_series, periods, freq=freq)
    try:
        cols = list(train_events.columns)
        is_weekly = freq.startswith("W")
        model = Prophet(
            weekly_seasonality=(not is_weekly),
            yearly_seasonality=(len(values) >= (104 if is_weekly else 365)),
            daily_seasonality=False,
        )
        for c in cols:
            model.add_regressor(c)

        train_df = pd.DataFrame({"ds": train_series.index, "y": values})
        for c in cols:
            train_df[c] = np.asarray(train_events[c].values, dtype=float)
        model.fit(train_df)

        future = pd.DataFrame({"ds": test_events.index})
        for c in cols:
            future[c] = np.asarray(test_events[c].values, dtype=float)
        return model.predict(future)["yhat"].to_numpy()[:periods]
    except Exception:
        return predict_prophet(train_series, periods, freq=freq)


def predict_tft(dated_series, periods, freq="D"):
    """Layer 1 고도화 모델(TFT, Temporal Fusion Transformer) — 논문용 Prophet 대비
    성능 비교 목적. 단일 품목 시계열에 대해 짧게 학습(few-epoch)한 뒤,
    **관측 구간 이후의 미래 periods 구간**을 예측한다. 데이터가 부족
    (encoder+prediction 구간 확보 불가)하거나 미설치·학습 실패 시
    predict_prophet으로 폴백한다."""
    values = np.asarray(
        dated_series.values if hasattr(dated_series, "values") else dated_series,
        dtype=float,
    )
    n = len(values)
    encoder_len = min(30, max(10, n - periods - 5))
    if not TFT_AVAILABLE or n < encoder_len + periods + 5:
        return predict_prophet(dated_series, periods, freq=freq)
    try:
        torch.manual_seed(42)
        df = pd.DataFrame(
            {
                "time_idx": np.arange(n),
                "group": "series",
                "value": values,
            }
        )

        # 학습은 관측된 전체 이력(0 ~ n-1)을 사용한다. 예측 대상이 '미래'이므로
        # 가장 최근 구간을 학습에서 떼어놓을 이유가 없다.
        training = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="value",
            group_ids=["group"],
            min_encoder_length=max(1, encoder_len // 2),
            max_encoder_length=encoder_len,
            min_prediction_length=1,
            max_prediction_length=periods,
            time_varying_unknown_reals=["value"],
            target_normalizer=GroupNormalizer(groups=["group"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        train_dataloader = training.to_dataloader(
            train=True, batch_size=16, num_workers=0
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=8,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(),
            optimizer="adam",
            log_interval=-1,
        )

        trainer = pl.Trainer(
            max_epochs=8,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(tft, train_dataloaders=train_dataloader)

        # 미래 구간 예측
        last_idx = int(df["time_idx"].iloc[-1])
        encoder_df = df[df["time_idx"] > last_idx - encoder_len]
        future_df = pd.concat(
            [
                df.iloc[[-1]].assign(time_idx=last_idx + i)
                for i in range(1, periods + 1)
            ],
            ignore_index=True,
        )
        predict_df = pd.concat([encoder_df, future_df], ignore_index=True)

        predict_ds = TimeSeriesDataSet.from_dataset(
            training, predict_df, predict=True, stop_randomization=True
        )
        predict_dl = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

        raw_predictions = tft.predict(predict_dl, mode="prediction")
        forecast = np.asarray(raw_predictions[0]).flatten()[:periods]
        if len(forecast) < periods:
            return predict_prophet(dated_series, periods, freq=freq)
        return forecast
    except Exception:
        return predict_prophet(dated_series, periods, freq=freq)


def predict_lstm(series, weeks=5):
    if not TF_AVAILABLE:
        return [0] * weeks
    win = 4
    if len(series) < win + 5:
        return predict_linear_trend_force(series, weeks)
    scaler = MinMaxScaler(feature_range=(0, 1))
    s_scaled = scaler.fit_transform(np.array(series).reshape(-1, 1))
    X, y = [], []
    for i in range(len(s_scaled) - win):
        X.append(s_scaled[i : i + win])
        y.append(s_scaled[i + win])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(win, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=30, verbose=0, batch_size=2)

    preds = []
    curr = s_scaled[-win:]
    for _ in range(weeks):
        pred_sc = model.predict(curr.reshape(1, win, 1), verbose=0)[0][0]
        preds.append(pred_sc)
        curr = np.append(curr[1:], [[pred_sc]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# 4. LangGraph 에이전트 (조회 → 예측 → 해석 → 브리핑)
class ForecastBriefState(TypedDict):
    item: str
    tier: str
    history: List[float]
    forecast: List[float]
    dates: List[str]
    recent_avg: float
    next_avg: float
    pct_change: float
    trend: str
    peak_week: str
    briefing: str
    shap_top: str


def _retrieve_node(state: ForecastBriefState) -> dict:
    """조회: 최근 8주 판매 이력만 추려 다음 단계로 전달."""
    history = state["history"]
    recent_history = history[-8:] if len(history) >= 8 else history
    return {"history": recent_history}


def _predict_node(state: ForecastBriefState) -> dict:
    """예측: Streamlit에서 이미 계산된 앙상블 예측치를 그대로 통과시킨다."""
    return {}


def _interpret_node(state: ForecastBriefState) -> dict:
    """해석: 최근 평균 대비 예측 평균의 변화율로 트렌드를 규정 (SHAP 기여도 분해의 임시 근사)."""
    history = state["history"]
    forecast = state["forecast"]
    recent_avg = float(np.mean(history)) if history else 0.0
    next_avg = float(np.mean(forecast)) if forecast else 0.0
    pct_change = ((next_avg - recent_avg) / recent_avg * 100) if recent_avg > 0 else 0.0

    if pct_change >= 10:
        trend = "증가"
    elif pct_change <= -10:
        trend = "감소"
    else:
        trend = "보합"

    peak_idx = int(np.argmax(forecast)) if forecast else 0
    peak_week = state["dates"][peak_idx] if state.get("dates") else ""

    return {
        "recent_avg": round(recent_avg, 1),
        "next_avg": round(next_avg, 1),
        "pct_change": round(pct_change, 1),
        "trend": trend,
        "peak_week": peak_week,
        "shap_top": state.get("shap_top", ""),
    }


def _brief_node(state: ForecastBriefState) -> dict:
    """브리핑: 해석 결과를 자연어 문장으로 변환 (LLM 미연결 상태의 템플릿 기반 생성)."""
    trend_phrase = {
        "증가": f"최근 평균 대비 {abs(state['pct_change'])}% 증가가 예상됩니다.",
        "감소": f"최근 평균 대비 {abs(state['pct_change'])}% 감소가 예상됩니다.",
        "보합": "최근 평균과 비슷한 수준을 유지할 것으로 예상됩니다.",
    }[state["trend"]]

    action_phrase = {
        "증가": "결품을 막기 위해 생산량을 여유 있게 준비하는 것을 권장합니다.",
        "감소": "과다생산·폐기를 줄이기 위해 생산량을 보수적으로 조정하는 것을 권장합니다.",
        "보합": "평소 생산 계획을 유지해도 무방합니다.",
    }[state["trend"]]

    briefing = (
        f"'{state['item']}'({state['tier']})은 향후 5주 평균 약 {state['next_avg']}개 판매가 예상되며, "
        f"{trend_phrase} 특히 {state['peak_week']} 주간에 가장 높은 수요가 예상됩니다. {action_phrase}"
    )
    if state.get("shap_top"):
        briefing += f"\n\n최근 2주 판매에 영향을 준 요인: {state['shap_top']}"
    llm_text = generate_briefing(state)
    if llm_text:
        briefing = llm_text
    return {"briefing": briefing}


@st.cache_resource
def build_briefing_agent():
    graph = StateGraph(ForecastBriefState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("predict", _predict_node)
    graph.add_node("interpret", _interpret_node)
    graph.add_node("brief", _brief_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "predict")
    graph.add_edge("predict", "interpret")
    graph.add_edge("interpret", "brief")
    graph.add_edge("brief", END)

    return graph.compile()


# 5. 이벤트 우선 게이팅 (Layer 2, H1) — 통합 이벤트 데이터셋 기반 Walk-forward 백테스트
@st.cache_data
def load_event_dataset(uploaded_file):
    return pd.read_csv(uploaded_file, parse_dates=["date"])


def build_event_dummies(daily):
    """일별 이벤트 플래그(요일·공휴일·방학·시즌·날씨)를 0/1 더미로 변환."""
    e = pd.DataFrame(index=daily.index)
    e["is_weekend"] = daily["is_weekend"].astype(float)
    e["is_holiday"] = daily["is_holiday"].astype(float)
    e["is_vacation"] = daily["is_vacation"].astype(float)
    e["is_rain"] = (daily["precip_type"] == "비").astype(float)
    e["is_snow"] = (daily["precip_type"] == "눈").astype(float)
    e["is_christmas"] = (daily["season_period"] == "크리스마스시즌").astype(float)
    e["is_suneung"] = (daily["season_period"] == "수능시즌").astype(float)
    e["is_valentine"] = (daily["season_period"] == "밸런타인시즌").astype(float)
    e["is_chuseok"] = (daily["season_period"] == "추석시즌").astype(float)
    return e


def fit_event_elasticity(qty, events):
    """이벤트별 수요 탄력도(β_k)를 추정한다.

    국소 기저치(14일 중심 이동중앙값) 대비 실제값의 상대 편차를 이벤트 더미에 회귀시켜,
    'ŷ_final = ŷ_base × (1 + Σ βₖ·eₖ)' 융합식의 βₖ를 구한다.
    """
    baseline = qty.rolling(14, center=True, min_periods=7).median()
    baseline = baseline.bfill().ffill()
    pct_dev = (qty - baseline) / baseline.replace(0, np.nan)
    pct_dev = pct_dev.fillna(0.0)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(events.values, pct_dev.values)
    return dict(zip(events.columns, reg.coef_))


def apply_event_gating(base_forecast, beta, test_events, tau):
    """이벤트 강도 점수가 임계치 τ를 넘는 날에만 이벤트 조정치를 예측에 반영(게이팅)."""
    cols = list(test_events.columns)
    event_score = test_events.values @ np.array([beta[c] for c in cols])
    gate_on = np.abs(event_score) >= tau
    multiplier = np.where(gate_on, 1 + event_score, 1.0)
    gated_forecast = np.clip(np.asarray(base_forecast) * multiplier, 0, None)
    return gated_forecast, event_score, gate_on


def compute_wape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - pred)) / denom * 100)


# 7. 실시간 이상탐지 — SPC 관리도 3종 + Isolation Forest
def rolling_3sigma_flags(series, window=28, min_periods=10, L=3.0, non_negative=True):
    """3시그마(Shewhart) 관리도 — detect.control_charts.shewhart_3sigma 래퍼."""
    return shewhart_3sigma(
        series, L=L, window=window, min_periods=min_periods, non_negative=non_negative
    )


def ewma_flags(series, lam=0.2, L=3, window=28, min_periods=10, non_negative=True):
    """EWMA 관리도 — detect.control_charts.ewma_chart 래퍼."""
    return ewma_chart(
        series,
        lam=lam,
        L=L,
        window=window,
        min_periods=min_periods,
        non_negative=non_negative,
    )


def cusum_flags(series, k=0.5, h=5.0, window=28, min_periods=10):
    """CUSUM 관리도 — detect.control_charts.cusum_chart 래퍼 (이번에 새로 추가)."""
    return cusum_chart(series, k=k, h=h, window=window, min_periods=min_periods)


# 8. SHAP 기여도 분석 (XAI)

SHAP_LABELS = {
    "is_weekend": "주말",
    "is_holiday": "공휴일",
    "is_vacation": "방학",
    "is_rain": "비",
    "is_snow": "눈",
    "is_christmas": "크리스마스시즌",
    "is_suneung": "수능시즌",
    "is_valentine": "밸런타인시즌",
    "is_chuseok": "추석시즌",
    "temperature": "기온",
    "trend": "추세",
}


def build_shap_features(daily):
    """이벤트 더미 + 기온 + 추세(day index)로 이루어진 SHAP 학습용 피처 행렬."""
    features = build_event_dummies(daily)
    features["temperature"] = daily["temperature"]
    features["trend"] = np.arange(len(daily))
    return features


def train_shap_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def explain_with_shap(model, X_test):
    """TreeExplainer로 SHAP 값과 기저값(base value)을 계산."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    base_value = float(np.ravel(explainer.expected_value)[0])
    return shap_values, base_value


def format_shap_briefing(item, date, base_value, contribs, pred, actual, top_n=5):
    """SHAP 기여도를 '기저 + 이벤트별 기여도' 형태의 자연어 문장으로 변환."""
    top = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(
        top_n
    )
    parts = []
    for name, v in top.items():
        if abs(v) < 0.05:
            continue
        label = SHAP_LABELS.get(name, name)
        parts.append(f"{label} {v:+.1f}개")
    detail = ", ".join(parts) if parts else "뚜렷한 이벤트 기여 없음"
    return (
        f"'{item}' {date} 예측치는 {pred:.1f}개(실제 {actual:.0f}개)로, "
        f"기저치 {base_value:.1f}개에 {detail}가 더해진 값입니다."
    )


# 10. What-if 채팅  — 키워드 기반 시나리오 파싱 + H1 게이팅 모델로 실계산

WHATIF_KEYWORD_MAP = {
    "is_rain": ["비", "우천", "강수"],
    "is_snow": ["눈", "폭설"],
    "is_holiday": ["공휴일", "연휴", "휴일"],
    "is_vacation": ["방학"],
    "is_weekend": ["주말", "토요일", "일요일"],
    "is_christmas": ["크리스마스", "성탄"],
    "is_suneung": ["수능"],
    "is_valentine": ["밸런타인", "발렌타인"],
    "is_chuseok": ["추석"],
}


def whatif_compute(beta, base_forecast, events, tau):
    """이벤트 조건을 반영한 예상 판매량을 계산한다.

    ŷ_final = ŷ_base x (1 + Σ βₖ·eₖ)  단, |Σ βₖ·eₖ| >= τ 일 때만 반영(게이팅).

    게이팅을 두는 이유는 약한 이벤트까지 매번 예측을 흔들면 오히려 정확도가 떨어지기
    때문이다. 다만 이 때문에 "비를 체크했는데 숫자가 그대로"인 상황이 생기므로,
    화면에서는 이벤트별 기여도와 미발동 사유를 함께 보여줘야 한다.
    """
    contribs = {k: beta.get(k, 0.0) for k, v in events.items() if v}
    score = sum(beta.get(k, 0.0) * v for k, v in events.items())
    gate_on = abs(score) >= tau
    final = float(
        np.clip(base_forecast * (1 + score) if gate_on else base_forecast, 0, None)
    )
    return {
        "score": score,
        "gate_on": gate_on,
        "final": final,
        "production": int(np.ceil(final * 1.1)),
        "pct": (
            (final - base_forecast) / base_forecast * 100 if base_forecast > 0 else 0.0
        ),
        "contribs": contribs,
    }


def whatif_reply_text(item, cond_str, base_forecast, r, tau):
    """계산 결과를 한 문단 답변으로 만든다 (LLM 미연동 시 사용하는 기본 답변)."""
    reply = (
        f"조건({cond_str})을 반영하면 '{item}' 예상 판매량은 "
        f"{base_forecast:.0f}개 → **{r['final']:.0f}개** ({r['pct']:+.1f}%)입니다. "
        f"안전재고 10%를 더한 권장 생산량은 **{r['production']}개**입니다."
    )
    if not r["gate_on"]:
        reply += (
            f" (이벤트 강도 {r['score']:+.2f}가 임계치 τ={tau:.2f}보다 작아 게이팅이 발동하지 않았고, "
            f"기저 예측을 그대로 유지했습니다. τ를 낮추면 반영됩니다.)"
        )
    return reply


def parse_whatif_keywords(text):
    """채팅 문구에서 이벤트 키워드를 인식해 0/1 플래그로 변환 (단순 키워드 매칭, NLU 아님)."""
    flags = {k: False for k in WHATIF_KEYWORD_MAP}
    for key, kws in WHATIF_KEYWORD_MAP.items():
        if any(kw in text for kw in kws):
            flags[key] = True
    return flags


# 3. 메인 UI 구성 — 왼쪽 메뉴 + 페이지 구조
_today = pd.Timestamp.today()
_today_str = f"{_today.year}년 {_today.month}월 {_today.day}일 {'월화수목금토일'[_today.weekday()]}요일"
_greeting = (
    "좋은 아침이에요"
    if _today.hour < 11
    else "좋은 오후예요" if _today.hour < 18 else "오늘도 수고하셨어요"
)

NAV_KEY = "bf_page"  # 현재 메뉴
DATA_KEY = "bf_data_path"  # 지금 쓰는 판매 데이터 파일 경로 (모든 메뉴가 공유)
DATA_NAME_KEY = "bf_data_name"
UPLOAD_ID_KEY = "bf_upload_id"

PAGE_TODAY, PAGE_FORECAST, PAGE_SALES = "오늘 현황", "수요 예측", "판매 분석"
PAGE_ALERT, PAGE_REASON, PAGE_ASSIST = "이상 알림", "예측 이유", "AI 도우미"
PAGE_DATA, PAGE_SETTINGS, PAGE_DEV = "데이터 관리", "설정", "모델 검증(개발용)"

# (메뉴 이름, 아이콘) — 아이콘은 Streamlit 내장 Material 아이콘
NAV_ITEMS = [
    (PAGE_TODAY, "dashboard"),
    (PAGE_FORECAST, "event_note"),
    (PAGE_SALES, "bar_chart"),
    (PAGE_ALERT, "notifications"),
    (PAGE_REASON, "manage_search"),
    (PAGE_ASSIST, "chat_bubble"),
    (PAGE_DATA, "database"),
    (PAGE_SETTINGS, "settings"),
    (PAGE_DEV, "science"),
]
NAV_ICON = dict(NAV_ITEMS)

# 업종별 용어: (예측 대상, 추천 단위)
BIZ_TERMS = {
    "베이커리": ("품목", "생산량"),
    "음식점": ("메뉴", "준비량"),
    "소매점": ("상품", "발주량"),
    "유통업": ("상품", "출고량"),
}
PROFILE_KEY = "bf_profile"


def _profile_path():
    return auth_store.user_data_dir(CURRENT_USER["id"]) / "profile.json"


def load_profile():
    """계정별 매장 프로필(매장 이름·업종) 불러오기."""
    if PROFILE_KEY not in st.session_state:
        prof = {"shop_name": CURRENT_USER["shop_name"], "biz_type": "베이커리"}
        try:
            prof.update(json.loads(_profile_path().read_text(encoding="utf-8")))
        except Exception:
            pass
        st.session_state[PROFILE_KEY] = prof
    return st.session_state[PROFILE_KEY]


def save_profile(prof):
    st.session_state[PROFILE_KEY] = prof
    try:
        _profile_path().write_text(
            json.dumps(prof, ensure_ascii=False), encoding="utf-8"
        )
        return True
    except Exception:
        return False


PROFILE = load_profile()
SHOP_NAME = PROFILE["shop_name"]
TARGET, UNIT = BIZ_TERMS.get(PROFILE["biz_type"], BIZ_TERMS["베이커리"])


def josa(word, pair):
    """받침에 맞는 조사 (예: josa('메뉴', '을/를') → '메뉴를')."""
    c = word[-1]
    has = "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 != 0
    a, b = pair.split("/")
    return word + (a if has else b)


# 페이지 상단 제목 (작은 제목, 큰 제목, 설명)
PAGE_META = {
    PAGE_FORECAST: (
        "판매·수요 예측",
        "오늘의 수요 예측",
        f'예상 판매량을 확인하고 {josa(UNIT, "을/를")} 정하세요. 표에서 {josa(TARGET, "을/를")} 누르면 5주 예측이 열려요.',
    ),
    PAGE_SALES: (
        "판매 인사이트",
        "판매 분석",
        "많이 팔리는 품목과 앞으로의 흐름을 봅니다.",
    ),
    PAGE_ALERT: (
        "매장 알림",
        "이상 알림",
        f'평소와 다르게 팔린 {josa(TARGET, "을/를")} 확인하고 {josa(UNIT, "을/를")} 조정하세요.',
    ),
    PAGE_REASON: (
        "예측 근거",
        "예측 이유",
        "날씨·공휴일·방학이 판매량에 준 영향을 봅니다.",
    ),
    PAGE_ASSIST: (
        "상황별 상담",
        "AI 도우미",
        "조건을 바꿔 보며 준비량이 어떻게 달라지는지 확인하세요.",
    ),
    PAGE_DATA: ("매장 데이터", "데이터 관리", "POS 판매내역을 올리고 관리합니다."),
    PAGE_SETTINGS: ("매장 관리", "설정", "매장과 계정 정보를 관리합니다."),
    PAGE_DEV: (
        "개발용",
        "모델 검증",
        "모델 성능을 검증하는 화면이에요. 매장 운영에는 필요하지 않아요.",
    ),
}


def go_to(page):
    """버튼 on_click 용: 다른 메뉴로 이동."""
    st.session_state[NAV_KEY] = page


def use_data(path, name):
    """모든 메뉴가 같이 쓸 판매 데이터 파일을 지정한다."""
    st.session_state[DATA_KEY] = str(path)
    st.session_state[DATA_NAME_KEY] = name


def start_with_sample():
    """버튼 on_click 용: 예시 데이터로 바꾸고 오늘 현황으로 이동."""
    use_data(SAMPLE_DATA_PATH, "예시 데이터")
    go_to(PAGE_TODAY)


# --- 사이드바: 매장 카드 + 메뉴 ---
with st.sidebar:
    bf_html(f"""
        <div class="bf-store">
          <div class="bf-store-icon">🏪</div>
          <div>
            <div class="bf-store-name">{SHOP_NAME}</div>
            <div class="bf-store-type">{PROFILE['biz_type']}</div>
          </div>
        </div>
        <div class="bf-nav-label">매장 운영</div>
        """)
    if NAV_KEY not in st.session_state:
        st.session_state[NAV_KEY] = PAGE_TODAY
    page = st.radio(
        "메뉴",
        [name for name, _ in NAV_ITEMS],
        key=NAV_KEY,
        format_func=lambda p: f":material/{NAV_ICON[p]}: {p}",
        label_visibility="collapsed",
    )

render_sidebar_account()

# --- 상단 바 + 페이지 제목 ---
bf_html(f"""
    <div class="bf-topbar">
      <span>🏪 {SHOP_NAME}</span><span class="bf-topbar-sep">/</span><span>{_today_str}</span>
    </div>
    """)
if page == PAGE_TODAY:
    bf_html(f"""
        <div class="bf-hero">
          <div>
            <div class="bf-hero-eyebrow">오늘의 운영 요약</div>
            <div class="bf-hero-title">{_greeting}</div>
            <div class="bf-hero-desc">오늘의 판매 예측과 {UNIT}을 확인하세요.</div>
          </div>
          <div class="bf-hero-chip">📅 {_today_str}</div>
        </div>
        """)
else:
    _eyebrow, _title, _desc = PAGE_META[page]
    bf_html(f"""
        <div class="bf-hero">
          <div>
            <div class="bf-hero-eyebrow">{_eyebrow}</div>
            <div class="bf-hero-title">{_title}</div>
            <div class="bf-hero-desc">{_desc}</div>
          </div>
        </div>
        """)


# --- 오늘 현황: 같은 요일 평균 기준 추천 생산량 ---
STATUS_STYLE = {
    "수요 증가": "up",
    "수요 감소": "down",
    "확인 필요": "warn",
    "정상": "normal",
}


def build_today_table(df, max_date, days_ahead=1):
    """
    품목별 '오늘(데이터 마지막 날의 다음 날)' 예상과 추천 생산량.
    - 오늘 예상: 지난 4주 같은 요일 평균
    - 평소 판매량: 지난 8주 같은 요일 평균 (요일 차이 때문에 생기는 착시를 막기 위해 같은 요일끼리 비교)
    - 추천 생산량: 오늘 예상 + 10% 여유 (이상 알림의 권장 준비량과 같은 기준)
    ※ 수요 예측 메뉴의 예측 모델은 건드리지 않는 별도 요약 계산이다.
    """
    target = (max_date + timedelta(days=days_ahead)).normalize()
    same_days = [target - timedelta(weeks=w) for w in range(1, 9)]  # 지난 8주 같은 요일
    recent = df[df["Date"].isin(same_days)]
    if recent.empty:
        return target, pd.DataFrame()

    wide = (
        recent.groupby(["Item", "Date"])["Quantity"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=same_days, fill_value=0.0)
    )
    expected = wide[same_days[:4]].mean(axis=1)
    usual = wide.mean(axis=1)

    t = pd.DataFrame({"평소": usual, "오늘 예상": expected})
    t = t[t["오늘 예상"] >= 0.5]  # 안 만드는 품목 제외
    t["추천"] = np.ceil(t["오늘 예상"] * 1.1).astype(int)
    t["변화"] = np.where(t["평소"] > 0, (t["오늘 예상"] / t["평소"] - 1) * 100, 0.0)
    t["상태"] = np.select(
        [t["변화"].abs() >= 50, t["변화"] >= 20, t["변화"] <= -20],
        ["확인 필요", "수요 증가", "수요 감소"],
        default="정상",
    )
    return target, t.sort_values("추천", ascending=False)


def render_today(df, max_date):
    target, t = build_today_table(df, max_date)
    if t.empty:
        st.info(
            "같은 요일 판매 기록이 부족해서 오늘 예상을 낼 수 없어요. 최소 4주 이상의 기록이 필요해요."
        )
        return

    total_exp, total_usual = t["오늘 예상"].sum(), t["평소"].sum()
    total_pct = (total_exp / total_usual - 1) * 100 if total_usual > 0 else 0.0
    n_up = int((t["상태"] == "수요 증가").sum())
    n_check = int(t["상태"].isin(["확인 필요", "수요 감소"]).sum())
    trend_word = (
        f"평소보다 {abs(total_pct):.0f}% {'많아요' if total_pct >= 0 else '적어요'}"
    )

    bf_html(f"""
        <div class="bf-kpi-grid">
          <div class="bf-kpi">
            <div class="bf-kpi-icon">📈</div>
            <div class="bf-kpi-label">오늘 예상 판매량</div>
            <div class="bf-kpi-value">{total_exp:,.0f}<small>개</small></div>
            <div class="bf-kpi-foot">{trend_word}</div>
          </div>
          <div class="bf-kpi">
            <div class="bf-kpi-icon">🧺</div>
            <div class="bf-kpi-label">오늘 준비할 {TARGET}</div>
            <div class="bf-kpi-value">{len(t)}<small>개</small></div>
            <div class="bf-kpi-foot">총 {t['추천'].sum():,}개 추천</div>
          </div>
          <div class="bf-kpi">
            <div class="bf-kpi-icon">⬆️</div>
            <div class="bf-kpi-label">수요 증가 예상</div>
            <div class="bf-kpi-value">{n_up}<small>개</small></div>
            <div class="bf-kpi-foot">{UNIT}을 늘려 보세요</div>
          </div>
          <div class="bf-kpi bf-kpi-alert">
            <div class="bf-kpi-icon">🔔</div>
            <div class="bf-kpi-label">확인 필요</div>
            <div class="bf-kpi-value">{n_check}<small>개</small></div>
            <div class="bf-kpi-foot">크게 늘거나 줄어든 품목</div>
          </div>
        </div>
        """)

    left, right = st.columns([1.75, 1], gap="large")
    with left:
        head_l, head_r = st.columns([3, 1.3])
        head_l.markdown(
            f'<div class="bf-section">오늘의 추천 {UNIT}</div>', unsafe_allow_html=True
        )
        head_r.button(
            "전체 수요 예측 →",
            on_click=go_to,
            args=(PAGE_FORECAST,),
            use_container_width=True,
        )

        rows = []
        for item, r in t.head(10).iterrows():
            arrow = "↗" if r["변화"] > 0 else ("↘" if r["변화"] < 0 else "→")
            chg_cls = "bf-up" if r["변화"] > 0 else ("bf-down" if r["변화"] < 0 else "")
            rows.append(
                f"<tr><td class='bf-td-item'>{item}</td>"
                f"<td>{r['평소']:.0f}개</td><td>{r['오늘 예상']:.0f}개</td>"
                f"<td class='bf-td-rec'>{r['추천']}개</td>"
                f"<td class='{chg_cls}'>{arrow} {r['변화']:+.0f}%</td>"
                f"<td><span class='bf-badge bf-badge-{STATUS_STYLE[r['상태']]}'>{r['상태']}</span></td></tr>"
            )
        bf_html(f"""
            <table class="bf-table">
              <thead><tr><th>{TARGET}</th><th>평소</th><th>오늘 예상</th>
              <th class="bf-th-rec">추천 {UNIT}</th><th>변화</th><th>상태</th></tr></thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
            """)
        st.caption(
            f"{target:%m월 %d일}({'월화수목금토일'[target.weekday()]}) 기준 · "
            "오늘 예상은 지난 4주 같은 요일 평균, 평소는 지난 8주 같은 요일 평균, "
            f"추천 {UNIT}은 예상보다 10% 넉넉하게 잡은 수예요."
        )
        if len(t) > 10:
            with st.expander(f"전체 {TARGET} 보기 ({len(t)}개)"):
                full = t.copy()
                full["평소"] = full["평소"].round(0)
                full["오늘 예상"] = full["오늘 예상"].round(0)
                full["변화"] = full["변화"].map(lambda v: f"{v:+.0f}%")
                st.dataframe(full.rename_axis(TARGET), use_container_width=True)

    with right:
        st.markdown(
            f'<div class="bf-section">오늘 확인할 {TARGET}</div>',
            unsafe_allow_html=True,
        )
        watch = t[t["상태"] != "정상"].copy()
        watch = watch.reindex(
            watch["변화"].abs().sort_values(ascending=False).index
        ).head(4)
        if watch.empty:
            st.success("오늘은 평소와 크게 다른 품목이 없어요.")
        for item, r in watch.iterrows():
            msg = (
                "평소보다 많이 팔릴 것 같아요. 넉넉히 준비하세요."
                if r["변화"] > 0
                else f"평소보다 덜 팔리고 있어요. {UNIT}을 줄여 보세요."
            )
            bf_html(f"""
                <div class="bf-watch bf-watch-{STATUS_STYLE[r['상태']]}">
                  <div class="bf-watch-head">
                    <span class="bf-watch-name">{item}</span>
                    <span class="bf-badge bf-badge-{STATUS_STYLE[r['상태']]}">{r['상태']}</span>
                  </div>
                  <div class="bf-watch-num">평소 {r['평소']:.0f}개 → 오늘 예상 <b>{r['오늘 예상']:.0f}개</b></div>
                  <div class="bf-watch-msg">{msg}</div>
                </div>
                """)
        st.button(
            "이상 알림 자세히 보기",
            on_click=go_to,
            args=(PAGE_ALERT,),
            use_container_width=True,
        )


@st.cache_data(show_spinner=False)
def real_shap_summary(path, item, horizon=14, top_n=3):
    """브리핑용: 최근 2주 SHAP 평균 기여도 상위 요인 (날씨·행사 열이 없으면 빈 문자열)."""
    if not (ML_AVAILABLE and SHAP_AVAILABLE):
        return ""
    sdf = load_event_dataset(path)
    need = {
        "date",
        "item",
        "sales_qty",
        "is_weekend",
        "is_holiday",
        "is_vacation",
        "season_period",
        "precip_type",
        "temperature",
    }
    if need - set(sdf.columns):
        return ""
    d = (
        sdf[sdf["item"] == item]
        .groupby("date")
        .agg(
            sales_qty=("sales_qty", "sum"),
            is_weekend=("is_weekend", "first"),
            is_holiday=("is_holiday", "first"),
            is_vacation=("is_vacation", "first"),
            season_period=("season_period", "first"),
            precip_type=("precip_type", "first"),
            temperature=("temperature", "first"),
        )
        .sort_index()
        .asfreq("D")
    )
    d["sales_qty"] = d["sales_qty"].fillna(0.0)
    d = d.ffill()
    if len(d) < horizon + 60:
        return ""
    X, y = build_shap_features(d), d["sales_qty"]
    model = train_shap_model(X.iloc[:-horizon], y.iloc[:-horizon])
    sv, _ = explain_with_shap(model, X.iloc[-horizon:])
    mean = pd.Series(np.asarray(sv).mean(axis=0), index=X.columns)
    top = mean.reindex(mean.abs().sort_values(ascending=False).index)[:top_n]
    return ", ".join(f"{SHAP_LABELS.get(k, k)} {v:+.1f}개" for k, v in top.items())


def chip_select(label, options, key, default=None):
    """칩 형태 단일 선택 (구버전 Streamlit은 가로 라디오로 대체)."""
    default = default or options[0]
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label, options, default=default, key=key)
        return v or default
    return st.radio(label, options, horizontal=True, key=key)


# --- 데이터 관리 ---
CSV_TEMPLATE = "date,item,sales_qty\n2026-10-01,소금빵,72\n2026-10-01,크루아상,55\n"


def render_data_page():
    data_path = st.session_state.get(DATA_KEY)
    with st.container(border=True):
        st.markdown("##### 지금 쓰는 판매 데이터")
        if data_path:
            cur = load_data(data_path)
            c1, c2, c3 = st.columns(3)
            c1.metric("파일", st.session_state.get(DATA_NAME_KEY, Path(data_path).name))
            c2.metric("판매 기록", f"{len(cur):,}건")
            c3.metric(
                "마지막 판매일",
                f"{cur['Date'].max():%Y-%m-%d}" if not cur.empty else "-",
            )
        else:
            st.info(
                "아직 불러온 데이터가 없어요. 아래에서 파일을 올리거나 예시 데이터로 시작하세요."
            )

    with st.container(border=True):
        st.markdown("##### 새 판매 데이터 올리기")
        st.caption(
            "POS에서 내려받은 판매내역 CSV를 올려 주세요. 꼭 필요한 열: date · item · sales_qty"
        )
        up = st.file_uploader(
            "판매내역 CSV",
            type=["csv"],
            label_visibility="collapsed",
            key="data_upload",
        )
        # 같은 파일을 화면이 다시 그려질 때마다 또 저장하지 않도록 한 번만 처리
        if up is not None and st.session_state.get(UPLOAD_ID_KEY) != up.file_id:
            raw_bytes = up.getvalue()
            try:
                saved_path = auth_store.user_data_dir(CURRENT_USER["id"]) / up.name
                saved_path.write_bytes(raw_bytes)
                auth_store.record_dataset(
                    CURRENT_USER["id"],
                    up.name,
                    saved_path,
                    n_rows=max(len(raw_bytes.splitlines()) - 1, 0),
                )
            except Exception as e:
                # 계정 폴더 저장이 안 되면 임시 파일로라도 이번 분석은 진행한다
                import tempfile

                saved_path = (
                    Path(tempfile.gettempdir()) / f"bf_{CURRENT_USER['id']}_{up.name}"
                )
                saved_path.write_bytes(raw_bytes)
                st.warning(f"계정에 저장하지 못해 이번에만 쓸 수 있어요. ({e})")
            load_data.clear()
            load_event_dataset.clear()
            use_data(saved_path, up.name)
            st.session_state[UPLOAD_ID_KEY] = up.file_id
            st.success(
                f"'{up.name}' 파일을 불러왔어요. 왼쪽 메뉴에서 오늘 현황을 확인해 보세요."
            )

    c_left, c_right = st.columns(2)
    with c_left:
        with st.container(border=True):
            st.markdown("##### 전에 올린 파일")
            my_datasets = auth_store.list_datasets(CURRENT_USER["id"])
            if not my_datasets:
                st.caption("아직 올린 파일이 없어요.")
            else:
                options = {
                    f"{d['filename']}  ({d['uploaded_at'][:16]})": d
                    for d in my_datasets
                }
                picked = st.selectbox(
                    "파일 선택", list(options.keys()), label_visibility="collapsed"
                )
                if st.button("이 파일 쓰기", use_container_width=True):
                    chosen = Path(options[picked]["stored_path"])
                    if chosen.exists():
                        use_data(chosen, options[picked]["filename"])
                        st.rerun()
                    else:
                        st.warning("파일을 찾을 수 없어요. 다시 올려 주세요.")
    with c_right:
        with st.container(border=True):
            st.markdown("##### 실제 데이터가 아직 없나요?")
            st.download_button(
                "CSV 양식 받기",
                data=CSV_TEMPLATE.encode("utf-8-sig"),
                file_name="판매내역_양식.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if SAMPLE_DATA_PATH.exists():
                st.button(
                    "예시 데이터로 보기",
                    type="primary",
                    use_container_width=True,
                    on_click=start_with_sample,
                )


# --- 설정 ---
def render_settings_page():
    prof = dict(PROFILE)
    with st.container(border=True):
        st.markdown("##### 🏪 매장 프로필")
        name = st.text_input("매장 이름", value=prof["shop_name"], key="set_name")
        st.markdown("**업종**")
        cols = st.columns(4)
        if "set_biz" not in st.session_state:
            st.session_state["set_biz"] = prof["biz_type"]
        for col, (biz, (tgt, unit)) in zip(cols, BIZ_TERMS.items()):
            picked = st.session_state["set_biz"] == biz
            if col.button(
                f"{biz}\n\n{tgt} · 추천 {unit}",
                key=f"biz_{biz}",
                type="primary" if picked else "secondary",
                use_container_width=True,
            ):
                st.session_state["set_biz"] = biz
                st.rerun()
        tgt, unit = BIZ_TERMS[st.session_state["set_biz"]]
        bf_html(f"""
            <div class="bf-sum">
              <div>업종<b>{st.session_state['set_biz']}</b></div>
              <div>예측 대상<b>{tgt}</b></div>
              <div>추천 단위<b>{unit}</b></div>
            </div>
            """)
        if st.button("저장", type="primary"):
            ok = save_profile(
                {
                    "shop_name": name.strip() or prof["shop_name"],
                    "biz_type": st.session_state["set_biz"],
                }
            )
            st.toast(
                "저장했어요."
                if ok
                else "이번 접속에만 적용돼요. (계정 폴더에 저장하지 못했어요)"
            )
            st.rerun()
        st.caption(
            f"업종을 바꾸면 화면의 '품목·생산량' 같은 용어가 '{tgt}·{unit}'처럼 바뀌어요."
        )

    def _open_account():
        st.session_state[ACCOUNT_VIEW_KEY] = True

    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown("##### 👤 계정")
        c1.caption("비밀번호와 로그인 정보를 바꿀 수 있어요.")
        c2.button("계정 관리", on_click=_open_account, use_container_width=True)
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown("##### 🗂 판매 데이터")
        c1.caption("POS 판매내역을 올리거나 예시 데이터로 바꿀 수 있어요.")
        c2.button(
            "데이터 관리", on_click=go_to, args=(PAGE_DATA,), use_container_width=True
        )


# --- 데이터가 없을 때 첫 화면 ---
def render_empty_state():
    bf_html("""
        <div class="bf-empty">
          <div class="bf-empty-icon">📂</div>
          <div class="bf-empty-title">먼저 판매 데이터를 불러와 주세요</div>
          <div class="bf-empty-desc">POS에서 받은 판매내역(CSV)을 올리면 오늘 만들 양을 바로 계산해 드려요.<br>
          파일이 없으면 예시 데이터로 먼저 둘러볼 수 있어요.</div>
        </div>
        """)
    _, c1, c2, _ = st.columns([1, 1.2, 1.2, 1])
    if SAMPLE_DATA_PATH.exists():
        c1.button(
            "예시 데이터로 보기",
            type="primary",
            use_container_width=True,
            on_click=start_with_sample,
        )
    c2.button(
        "판매 데이터 올리기",
        on_click=go_to,
        args=(PAGE_DATA,),
        use_container_width=True,
    )


uploaded_file = st.session_state.get(DATA_KEY)

if page == PAGE_DATA:
    render_data_page()
elif page == PAGE_SETTINGS:
    render_settings_page()
elif uploaded_file is None:
    render_empty_state()
else:
    df = load_data(uploaded_file)
    if df.empty:
        st.error(
            "파일을 읽지 못했어요. date, item, sales_qty 열이 있는지 확인해 주세요."
        )
    else:
        max_date = df["Date"].max()

        # [전처리] 품목 등급 산출
        tier_df = calculate_item_tiers(df)
        # 등급 정보를 메인 데이터프레임과 병합할 수도 있지만, 필터링용 리스트로 활용

        if page == PAGE_TODAY:
            render_today(df, max_date)

        # 수요 예측: 조건 + 추천표 + 선택 품목 5주 예측
        if page == PAGE_FORECAST:
            with st.container(border=True):
                f1, f2 = st.columns([1, 2])
                with f1:
                    day_opt = chip_select("날짜", ["오늘", "내일"], key="fc_day")
                with f2:
                    activity_opt = chip_select(
                        "최근 판매",
                        ["전체", "최근 3개월", "최근 6개월", "최근 1년"],
                        key="fc_recent",
                    )
                f3, f4 = st.columns([1, 2])
                tier_options = [
                    "전체",
                    "시그니처 (상위 10%)",
                    "인기 (상위 30%)",
                    "스테디셀러 (상위 60%)",
                    "일반",
                ]
                tier_opt = f3.selectbox(
                    "판매량 순위", tier_options, index=0, key="fc_tier"
                )
                search_txt = f4.text_input(
                    f"{TARGET} 검색", placeholder="예: 식빵", key="fc_search"
                )

            # 기간·순위 필터
            last_dates = df.groupby("Item")["Date"].max().reset_index()
            days = {"최근 3개월": 90, "최근 6개월": 180, "최근 1년": 365}.get(
                activity_opt
            )
            if days:
                active_item_list = last_dates[
                    last_dates["Date"] >= max_date - timedelta(days=days)
                ]["Item"].tolist()
            else:
                active_item_list = last_dates["Item"].tolist()
            if tier_opt != "전체":
                tier_item_list = tier_df[tier_df["Tier"] == tier_opt]["Item"].tolist()
            else:
                tier_item_list = tier_df["Item"].tolist()
            final_item_list = sorted(set(active_item_list) & set(tier_item_list))
            display_list = (
                [c for c in final_item_list if search_txt in c]
                if search_txt
                else final_item_list
            )

            # 추천표 (같은 요일 평균 기준)
            target, ft = build_today_table(
                df, max_date, days_ahead=1 if day_opt == "오늘" else 2
            )
            ft = ft[ft.index.isin(display_list)] if not ft.empty else ft
            selected_item = None

            if ft.empty:
                st.warning(
                    f"해당하는 {josa(TARGET, '이/가')} 없어요. 조건을 바꿔 보세요."
                )
            else:
                tier_map = tier_df.set_index("Item")["Tier"].str.split(" ").str[0]
                show = pd.DataFrame(
                    {
                        TARGET: ft.index,
                        "순위": ft.index.map(tier_map),
                        "평소": ft["평소"].round(0).astype(int).map(lambda v: f"{v}개"),
                        f"{day_opt} 예상": ft["오늘 예상"]
                        .round(0)
                        .astype(int)
                        .map(lambda v: f"{v}개"),
                        f"추천 {UNIT}": ft["추천"].map(lambda v: f"{v}개"),
                        "변화": ft["변화"].map(
                            lambda v: f"{'↗' if v > 0 else '↘' if v < 0 else '→'} {v:+.0f}%"
                        ),
                        "상태": ft["상태"],
                    }
                ).reset_index(drop=True)

                h1, h2, h3 = st.columns([3, 1, 1.2])
                h1.markdown(
                    f'<div class="bf-section">{TARGET}별 추천 {UNIT}</div>',
                    unsafe_allow_html=True,
                )
                h2.caption(
                    f"총 {len(show)}개 · {target:%m/%d}({'월화수목금토일'[target.weekday()]})"
                )
                h3.download_button(
                    "엑셀로 내보내기",
                    data=to_excel(show),
                    file_name=f"추천{UNIT}_{target:%Y%m%d}.xlsx",
                    use_container_width=True,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                badge = {
                    "수요 증가": "background-color:#dcfce7;color:#166534;font-weight:700",
                    "수요 감소": "background-color:#fee2e2;color:#991b1b;font-weight:700",
                    "확인 필요": "background-color:#ffedd5;color:#9a3412;font-weight:700",
                    "정상": "color:#334155",
                }
                smap = "map" if hasattr(show.style, "map") else "applymap"
                styled = getattr(show.style, smap)(
                    lambda v: badge.get(v, ""), subset=["상태"]
                )
                styled = getattr(styled, smap)(
                    lambda v: "background-color:#eff6ff;color:#2563eb;font-weight:800",
                    subset=[f"추천 {UNIT}"],
                )
                ev = st.dataframe(
                    styled,
                    hide_index=True,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="fc_table",
                    height=min(38 * (len(show) + 1) + 3, 460),
                )
                st.caption(
                    f"{day_opt} 예상: 지난 4주 같은 요일 평균 · 평소: 지난 8주 같은 요일 평균 · "
                    f"추천: 예상 +10% · 행을 누르면 아래에 5주 예측이 열려요."
                )
                rows = ev.selection.rows if ev is not None else []
                selected_item = show.iloc[rows[0]][TARGET] if rows else None

            # --- [분석 실행] ---
            if selected_item:
                st.divider()
                st.markdown(f"### {selected_item} · 앞으로 5주")

                # 품목 등급 표시
                item_tier_info = tier_df[tier_df["Item"] == selected_item][
                    "Tier"
                ].values[0]
                st.caption(f"판매량 순위: {item_tier_info}")

                item_df = df[df["Item"] == selected_item].sort_values("Date")
                item_weekly = (
                    item_df.set_index("Date").resample("W-MON")["Quantity"].sum()
                )

                start_date = item_weekly[item_weekly > 0].index.min()
                if pd.isna(start_date):
                    start_date = item_weekly.index.min()

                full_idx = pd.date_range(start=start_date, end=max_date, freq="W-MON")
                item_weekly = item_weekly.reindex(full_idx, fill_value=0)
                series_data = item_weekly.values

                with st.spinner("계산하고 있어요..."):
                    p_linear = predict_linear_trend_force(series_data, 5)
                    p_holt = predict_holt_trend(series_data, 5)
                    p_arima = predict_arima_trend(series_data, 5)
                    p_prophet = (
                        predict_prophet(item_weekly, 5, freq="W-MON")
                        if PROPHET_AVAILABLE
                        else [0] * 5
                    )
                    p_tft = (
                        predict_tft(item_weekly, 5, freq="W-MON")
                        if TFT_AVAILABLE
                        else [0] * 5
                    )

                    p_rf = predict_rf(series_data, 5) if ML_AVAILABLE else [0] * 5
                    p_xgb = predict_xgboost(series_data, 5) if ML_AVAILABLE else [0] * 5
                    p_lstm = predict_lstm(series_data, 5) if TF_AVAILABLE else [0] * 5

                    valid_preds = [p_linear, p_holt, p_arima]
                    if PROPHET_AVAILABLE:
                        valid_preds.append(p_prophet)
                    if TFT_AVAILABLE:
                        valid_preds.append(p_tft)
                    if ML_AVAILABLE:
                        valid_preds.extend([p_rf, p_xgb])
                    if TF_AVAILABLE:
                        valid_preds.append(p_lstm)

                    ens_pred = np.mean(valid_preds, axis=0)
                    ens_pred = [round(max(0, x), 1) for x in ens_pred]

                dates_str = [
                    (max_date + timedelta(weeks=i)).strftime("%Y-%m-%d")
                    for i in range(1, 6)
                ]

                res_df = pd.DataFrame(
                    {
                        "날짜": dates_str,
                        "앙상블(최종)": ens_pred,
                        "선형추세": [round(max(0, x), 1) for x in p_linear],
                        "Holt": [round(max(0, x), 1) for x in p_holt],
                        "Prophet": [round(max(0, x), 1) for x in p_prophet],
                        "TFT": [round(max(0, x), 1) for x in p_tft],
                    }
                )

                st.dataframe(
                    res_df.set_index("날짜").rename(
                        columns={"앙상블(최종)": "예상 판매량", "선형추세": "단순 추세"}
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "'예상 판매량'은 여러 예측 방법의 평균이에요. 오른쪽 열은 방법별 참고값입니다."
                )
                st.download_button(
                    "엑셀로 받기",
                    data=to_excel(res_df),
                    file_name=f"{selected_item}_예측.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                st.markdown("---")
                st.subheader("판매 흐름")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=item_weekly.index[-20:],
                        y=item_weekly.values[-20:],
                        name="실제 판매",
                        line=dict(color=BF_GRAY, dash="dot"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(res_df["날짜"]),
                        y=res_df["앙상블(최종)"],
                        name="예상 판매",
                        line=dict(color=BF_BLUE, width=4),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(res_df["날짜"]),
                        y=res_df["선형추세"],
                        name="단순 추세",
                        line=dict(color=BF_SKY, width=1, dash="dot"),
                    )
                )
                fig.update_layout(
                    height=500,
                    hovermode="x unified",
                    title=f"{selected_item} 주간 판매량 (최근 20주 + 앞으로 5주)",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("한 줄 요약")
                if LANGGRAPH_AVAILABLE:
                    with st.spinner("요약을 만들고 있어요..."):
                        agent = build_briefing_agent()
                        agent_state = agent.invoke(
                            {
                                "item": selected_item,
                                "tier": item_tier_info,
                                "history": [float(v) for v in item_weekly.values],
                                "forecast": [float(v) for v in ens_pred],
                                "dates": dates_str,
                                "shap_top": real_shap_summary(
                                    uploaded_file, selected_item
                                ),
                            }
                        )
                    st.chat_message("assistant").write(agent_state["briefing"])
                else:
                    st.caption("요약 기능이 꺼져 있어요. (langgraph 미설치)")

            elif not ft.empty:
                st.info(
                    f"표에서 {josa(TARGET, '을/를')} 누르면 앞으로 5주 예측과 요약이 여기에 나와요."
                )

        # 탭 2: 베스트셀러 TOP N
        if page == PAGE_SALES:
            top_n = st.radio(
                "상위 몇 개까지 볼까요?",
                [10, 20, 30],
                horizontal=True,
                format_func=lambda n: f"{n}개",
            )

            if st.button("예측 보기", type="primary"):
                top_items = (
                    df.groupby("Item")["Quantity"].sum().nlargest(top_n).index.tolist()
                )
                results = []
                dates_str = [
                    (max_date + timedelta(weeks=i)).strftime("%Y-%m-%d")
                    for i in range(1, 6)
                ]

                bar = st.progress(0)
                for idx, item in enumerate(top_items):
                    i_df = df[df["Item"] == item].sort_values("Date")
                    i_weekly = (
                        i_df.set_index("Date").resample("W-MON")["Quantity"].sum()
                    )
                    start_d = i_weekly[i_weekly > 0].index.min()
                    if pd.isna(start_d):
                        start_d = i_weekly.index.min()
                    i_series = i_weekly.reindex(
                        pd.date_range(start=start_d, end=max_date, freq="W-MON"),
                        fill_value=0,
                    ).values

                    p1 = predict_linear_trend_force(i_series, 5)
                    p2 = predict_holt_trend(i_series, 5)
                    avg = np.mean([p1, p2], axis=0)

                    row = {"품목명": item}
                    for i, d in enumerate(dates_str):
                        row[f"{i + 1}주차 ({d})"] = round(max(0, avg[i]), 1)
                    results.append(row)
                    bar.progress((idx + 1) / top_n)

                top_df = pd.DataFrame(results)
                st.dataframe(top_df, use_container_width=True)
                st.download_button(
                    "엑셀로 받기",
                    data=to_excel(top_df),
                    file_name=f"Top{top_n}_예측.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                st.markdown("---")
                if st.toggle("그래프로 보기", value=True):
                    plot_df = top_df.melt(
                        id_vars=["품목명"], var_name="주차", value_name="예측수량"
                    )
                    fig2 = px.line(
                        plot_df,
                        x="주차",
                        y="예측수량",
                        color="품목명",
                        markers=True,
                        title=f"상위 {top_n}개 품목 주간 예상 판매량",
                    )
                    fig2.update_layout(height=600, hovermode="x unified")
                    st.plotly_chart(fig2, use_container_width=True)

        # 탭 3: 판매 이상 알림 (재고 판단용)
        if page == PAGE_ALERT:
            st.caption(
                "평소보다 유독 많이 또는 적게 팔린 날이에요. "
                "'평소'는 지난 4주 같은 요일 평균이에요. 토요일은 토요일끼리 비교해요."
            )

            alert_file = uploaded_file  # 데이터 관리에서 불러온 파일을 함께 사용

            if alert_file is None:
                st.info("판매내역을 올리면 알림을 볼 수 있어요.")
            else:
                adf = load_event_dataset(alert_file)
                missing_cols = {"date", "item", "sales_qty"} - set(adf.columns)

                if missing_cols:
                    st.warning(
                        f"이 기능에는 날씨·행사 정보가 담긴 판매 파일이 필요해요. 빠진 열: {', '.join(sorted(missing_cols))}"
                    )
                elif not ML_AVAILABLE:
                    st.warning(
                        "scikit-learn이 설치되어 있지 않아 알림을 만들 수 없어요."
                    )
                else:
                    daily_total = (
                        adf.groupby("date")["sales_qty"].sum().asfreq("D").fillna(0.0)
                    )
                    all_items = sorted(adf["item"].unique())
                    by_item = {
                        it: adf[adf["item"] == it]
                        .groupby("date")["sales_qty"]
                        .sum()
                        .reindex(daily_total.index, fill_value=0.0)
                        for it in all_items
                    }

                    if len(daily_total) < 40:
                        st.warning("40일 이상의 판매 기록이 있어야 해요.")
                    else:
                        period = st.radio(
                            "확인할 기간",
                            [30, 60, 90],
                            index=0,
                            horizontal=True,
                            format_func=lambda d: f"{d}일",
                            key="alert_period",
                        )

                        with st.spinner("살펴보고 있어요..."):
                            # 시즌·공휴일·날씨를 알면 '왜 그랬는지'까지 말해줄 수 있다.
                            context = build_context(adf)
                            alerts = build_alerts(
                                by_item,
                                daily_total,
                                all_items,
                                recent_days=period,
                                context=context,
                            )
                            trends = {
                                it: trend_summary(by_item[it]) for it in all_items
                            }

                        n_high = sum(1 for a in alerts if a["severity"] == "높음")
                        n_short = sum(1 for a in alerts if a["pct"] > 0)
                        n_over = sum(1 for a in alerts if a["pct"] < 0)

                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            "평소와 다른 날",
                            f"{len(alerts)}건",
                            help=f"최근 {period}일 중 평소보다 20% 넘게 차이 난 날",
                        )
                        m2.metric(
                            "많이 팔린 날", f"{n_short}건", help="품절에 주의하세요"
                        )
                        m3.metric(
                            "적게 팔린 날", f"{n_over}건", help="재고가 남을 수 있어요"
                        )

                        # 알림 카드 (전체·급증·급감 + 상세보기)
                        st.markdown("#### 최근 알림")
                        if not alerts:
                            st.success(f"최근 {period}일 동안 크게 튄 날이 없었어요.")
                        else:
                            kind = chip_select(
                                "보기",
                                [
                                    f"전체 {len(alerts)}",
                                    f"급증 {n_short}",
                                    f"급감 {n_over}",
                                ],
                                key="alert_kind",
                            )
                            shown = [
                                a
                                for a in alerts
                                if kind.startswith("전체")
                                or (kind.startswith("급증") and a["pct"] > 0)
                                or (kind.startswith("급감") and a["pct"] < 0)
                            ]

                            for i, a in enumerate(shown[:20]):
                                up = a["pct"] > 0
                                s_item = by_item[a["item"]]
                                d = pd.Timestamp(a["date"])
                                weeks = [
                                    d - timedelta(weeks=w) for w in range(5, -1, -1)
                                ]
                                vals = s_item.reindex(weeks)
                                usual, actual = (
                                    vals.iloc[:-1].tail(4).mean(),
                                    vals.iloc[-1],
                                )
                                diff = (
                                    (actual / usual - 1) * 100
                                    if usual and usual > 0
                                    else 0
                                )
                                tag = (
                                    "이유 있음"
                                    if a["is_event"]
                                    else {
                                        "높음": "꼭 확인",
                                        "보통": "참고",
                                        "낮음": "작은 차이",
                                    }[a["severity"]]
                                )

                                with st.container(border=True):
                                    c1, c2 = st.columns([5, 1])
                                    c1.markdown(
                                        f"<div class='bf-al-meta bf-al-{'up' if up else 'down'}'>"
                                        f"{'📈 판매 급증' if up else '📉 판매 급감'}"
                                        f"<span>{d:%Y.%m.%d}({'월화수목금토일'[d.weekday()]}) · {tag}</span></div>"
                                        f"<div class='bf-al-name'>{a['item']}</div>"
                                        f"<div>평소 같은 요일보다 {abs(diff):.0f}% {'많이' if diff >= 0 else '적게'} 팔렸어요.</div>"
                                        f"<div class='bf-al-num'>평소 <b>{usual:.0f}개</b> → 실제 <b>{actual:.0f}개</b></div>",
                                        unsafe_allow_html=True,
                                    )
                                    open_key = f"al_open_{a['item']}_{d:%Y%m%d}"
                                    is_open = st.session_state.get(open_key, False)
                                    if c2.button(
                                        "접기 ▴" if is_open else "상세보기 ▾",
                                        key=f"btn_{open_key}",
                                        use_container_width=True,
                                    ):
                                        st.session_state[open_key] = not is_open
                                        st.rerun()
                                    if is_open:
                                        why = a["cause"] or a["meaning"]
                                        st.markdown(f"**왜 그랬을까요?**  \n{why}")
                                        st.markdown(
                                            f"**이렇게 해 보세요**  \n{a['action']}"
                                        )
                                        st.caption(a["detail"])
                                        figb = go.Figure(
                                            go.Bar(
                                                x=[f"{w:%m/%d}" for w in weeks],
                                                y=vals.fillna(0).values,
                                                marker_color=[BF_GRAY] * 5
                                                + ["#22C55E" if up else BF_RED],
                                            )
                                        )
                                        figb.update_layout(
                                            height=200,
                                            margin=dict(t=30, b=10, l=10, r=10),
                                            title=dict(
                                                text="최근 6주 같은 요일 판매",
                                                font=dict(size=13),
                                            ),
                                        )
                                        st.plotly_chart(
                                            figb,
                                            use_container_width=True,
                                            key=f"bar_{open_key}",
                                        )

                            if len(shown) > 20:
                                st.caption(f"외 {len(shown) - 20}건")

                        # ---------- 2. 요즘 추세 ----------
                        st.markdown("#### 요즘 흐름")
                        st.caption(
                            "하루 튄 것보다 몇 주째 이어지는 변화가 준비량을 정할 때 더 중요해요."
                        )

                        trend_rows = []
                        for it, t in trends.items():
                            if t is None:
                                continue
                            trend_rows.append(
                                {
                                    "품목": it,
                                    "방향": (
                                        "▲ 늘고 있음"
                                        if t["direction"] == "상승"
                                        else "▼ 줄고 있음"
                                    ),
                                    "최근 4주 변화": f"{t['change'] * 100:+.0f}%",
                                    "추천": t["action"],
                                }
                            )

                        if trend_rows:
                            trend_df = pd.DataFrame(trend_rows).sort_values(
                                "최근 4주 변화",
                                key=lambda s: s.str.rstrip("%").astype(float),
                                ascending=False,
                            )
                            st.dataframe(
                                trend_df, use_container_width=True, hide_index=True
                            )
                        else:
                            st.info("최근 4주 동안 눈에 띄게 늘거나 준 품목은 없어요.")

                        # ---------- 3. 요일별 준비량 ----------
                        st.markdown("#### 요일별 준비량")
                        st.caption(
                            "지난 4주 판매를 요일별로 정리했어요. 내일 준비량을 정할 때 참고하세요."
                        )

                        guide_item = st.selectbox(
                            "품목 선택", all_items, key="guide_item"
                        )
                        guide = weekday_guide(by_item[guide_item])

                        if guide is None:
                            st.info("판매 기록이 부족해서 요일별로 나눠 보기 어려워요.")
                        else:
                            gc1, gc2 = st.columns([1, 1.4])
                            with gc1:
                                st.dataframe(
                                    guide, use_container_width=True, hide_index=True
                                )
                                st.caption(
                                    "권장 준비량은 평균보다 10% 넉넉하게 잡은 수예요. "
                                    "남는 게 부담되면 평균에 맞추세요."
                                )
                            with gc2:
                                # 최근 60일 판매 흐름 + 평소 범위
                                s = by_item[guide_item].tail(60)
                                _, exp = weekday_residual(by_item[guide_item])
                                exp = exp.reindex(s.index)

                                figg = go.Figure()
                                figg.add_trace(
                                    go.Scatter(
                                        x=s.index,
                                        y=exp * 1.3,
                                        name="평소 범위(위)",
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo="skip",
                                    )
                                )
                                figg.add_trace(
                                    go.Scatter(
                                        x=s.index,
                                        y=exp * 0.7,
                                        name="평소 범위",
                                        line=dict(width=0),
                                        fill="tonexty",
                                        fillcolor="rgba(37,99,235,0.10)",
                                        hoverinfo="skip",
                                    )
                                )
                                figg.add_trace(
                                    go.Scatter(
                                        x=s.index,
                                        y=s,
                                        name="실제 판매량",
                                        line=dict(color=BF_NAVY, width=2),
                                    )
                                )

                                hit_dates = [
                                    a["date"]
                                    for a in alerts
                                    if a["item"] == guide_item and a["date"] in s.index
                                ]
                                if hit_dates:
                                    figg.add_trace(
                                        go.Scatter(
                                            x=hit_dates,
                                            y=s.loc[hit_dates],
                                            mode="markers",
                                            name="평소와 달랐던 날",
                                            marker=dict(
                                                color=BF_RED, size=10, symbol="circle"
                                            ),
                                        )
                                    )

                                figg.update_layout(
                                    height=330,
                                    hovermode="x unified",
                                    title=f"{guide_item} — 최근 60일",
                                    margin=dict(t=45, b=30, l=10, r=10),
                                    legend=dict(
                                        orientation="h", yanchor="bottom", y=1.0
                                    ),
                                )
                                st.plotly_chart(figg, use_container_width=True)

                        # ---------- 설명 ----------
                        with st.expander("알림은 이렇게 정해져요"):
                            st.markdown(
                                "**1. 요일을 맞춰 비교합니다.**  \n"
                                "빵집은 요일마다 판매량이 크게 다릅니다. 그래서 토요일은 직전 4주간의 "
                                "토요일 평균과만 비교합니다. 전체 평균과 비교하면 멀쩡한 주말이 매번 "
                                "'이상'으로 잡힙니다.\n\n"
                                "**2. 여러 방법으로 함께 확인합니다.**  \n"
                                "하루 크게 튄 날, 며칠에 걸쳐 서서히 밀린 날, 요일 패턴 자체가 깨진 날은 "
                                "각각 다른 방식으로 잡아야 놓치지 않습니다. 여러 방법이 함께 지목한 날만 "
                                "알림으로 올립니다.\n\n"
                                "**3. 20% 미만 차이는 걸러냅니다.**  \n"
                                "통계적으로는 차이가 있어도 현장에서 의미 없는 수준이면 알림을 만들지 "
                                "않습니다. 알림이 너무 잦으면 결국 아무도 안 보게 되기 때문입니다.\n\n"
                                "**4. 원인을 두 갈래로 나눕니다.**  \n"
                                "그 날 매장 전체 매출도 흔들렸다면 날씨·휴일 같은 바깥 요인일 가능성이 크고, "
                                "그 품목만 흔들렸다면 진열·품질·경쟁 제품 등 품목 자체의 문제일 가능성이 큽니다."
                            )

        # 탭 4: SHAP 기여도 분석
        if page == PAGE_REASON:
            st.caption(
                "날씨·공휴일·방학 같은 조건이 판매량을 얼마나 올리고 내렸는지 날짜별로 보여 줘요."
            )
            shap_file = uploaded_file  # 데이터 관리에서 불러온 파일을 함께 사용

            if shap_file is not None:
                sdf = load_event_dataset(shap_file)
                required_cols = {
                    "date",
                    "item",
                    "sales_qty",
                    "is_weekend",
                    "is_holiday",
                    "is_vacation",
                    "season_period",
                    "precip_type",
                    "temperature",
                }
                missing_cols = required_cols - set(sdf.columns)

                if missing_cols:
                    st.warning(
                        f"이 기능에는 날씨·행사 정보가 담긴 판매 파일이 필요해요. 빠진 열: {', '.join(sorted(missing_cols))}"
                    )
                elif not ML_AVAILABLE:
                    st.warning(
                        "scikit-learn이 설치되어 있지 않아 모델을 학습할 수 없습니다."
                    )
                elif not SHAP_AVAILABLE:
                    st.warning(
                        "shap이 설치되어 있지 않습니다. `pip install shap` 후 이용할 수 있습니다."
                    )
                else:
                    shap_items = sorted(sdf["item"].unique())
                    shap_item = st.selectbox("품목 선택", shap_items, key="shap_item")
                    shap_horizon = st.slider(
                        "최근 며칠을 볼까요?", 7, 60, 14, key="shap_horizon"
                    )

                    item_daily = (
                        sdf[sdf["item"] == shap_item]
                        .groupby("date")
                        .agg(
                            sales_qty=("sales_qty", "sum"),
                            is_weekend=("is_weekend", "first"),
                            is_holiday=("is_holiday", "first"),
                            is_vacation=("is_vacation", "first"),
                            season_period=("season_period", "first"),
                            precip_type=("precip_type", "first"),
                            temperature=("temperature", "first"),
                        )
                        .sort_index()
                        .asfreq("D")
                    )
                    item_daily["sales_qty"] = item_daily["sales_qty"].fillna(0.0)
                    item_daily = item_daily.ffill()

                    if len(item_daily) < shap_horizon + 60:
                        st.warning(
                            "판매 기록이 부족해요. (60일 + 보려는 기간만큼 필요)"
                        )
                    else:
                        X = build_shap_features(item_daily)
                        y = item_daily["sales_qty"]

                        X_train, X_test = X.iloc[:-shap_horizon], X.iloc[-shap_horizon:]
                        y_train, y_test = y.iloc[:-shap_horizon], y.iloc[-shap_horizon:]

                        with st.spinner("계산하고 있어요..."):
                            model = train_shap_model(X_train, y_train)
                            shap_values, base_value = explain_with_shap(model, X_test)
                            pred_test = model.predict(X_test)

                        st.markdown("#### 판매량에 영향을 많이 준 조건")
                        importance = pd.Series(
                            np.abs(shap_values).mean(axis=0), index=X.columns
                        ).sort_values(ascending=True)
                        importance.index = [
                            SHAP_LABELS.get(c, c) for c in importance.index
                        ]
                        fig6a = px.bar(
                            importance,
                            orientation="h",
                            labels={"value": "평균 영향 (개)", "index": "조건"},
                            title=f"{shap_item}",
                        )
                        st.plotly_chart(fig6a, use_container_width=True)

                        st.markdown("#### 날짜별로 보기")
                        sel_date = st.selectbox(
                            "날짜 선택",
                            list(X_test.index),
                            key="shap_date",
                            format_func=lambda d: d.strftime("%Y-%m-%d (%a)"),
                        )
                        idx = list(X_test.index).index(sel_date)
                        contribs = pd.Series(shap_values[idx], index=X.columns)
                        pred = pred_test[idx]
                        actual = y_test.iloc[idx]

                        briefing = format_shap_briefing(
                            shap_item,
                            sel_date.strftime("%Y-%m-%d"),
                            base_value,
                            contribs,
                            pred,
                            actual,
                        )
                        st.chat_message("assistant").write(briefing)

                        contrib_df = pd.DataFrame(
                            {
                                "조건": [SHAP_LABELS.get(c, c) for c in contribs.index],
                                "영향(개)": contribs.values,
                            }
                        )
                        contrib_df = contrib_df.reindex(
                            contrib_df["영향(개)"]
                            .abs()
                            .sort_values(ascending=False)
                            .index
                        )
                        fig6b = px.bar(
                            contrib_df,
                            x="영향(개)",
                            y="조건",
                            orientation="h",
                            color="영향(개)",
                            color_continuous_scale="RdBu",
                            color_continuous_midpoint=0,
                            title=f"{sel_date.strftime('%Y-%m-%d')} · 기본 {base_value:.1f}개에서 조건별로 더하고 뺀 값",
                        )
                        st.plotly_chart(fig6b, use_container_width=True)

                        with st.expander("표로 보기"):
                            st.dataframe(
                                contrib_df, use_container_width=True, hide_index=True
                            )
                            st.caption(
                                f"검증: 기저치({base_value:.1f}) + 전체 기여도 합({contribs.sum():.1f}) "
                                f"= 모델 예측치({pred:.1f})"
                            )
            else:
                st.info("날씨·행사 정보가 담긴 판매 CSV를 올리면 볼 수 있어요.")

        # 탭 5: What-if 채팅
        if page == PAGE_ASSIST:
            st.caption(
                "비가 오거나 공휴일이면 얼마나 팔릴까요? 조건을 고르면 예상 판매량과 준비량이 바로 바뀌어요. "
                "말로 물어봐도 돼요."
            )
            whatif_file = uploaded_file  # 데이터 관리에서 불러온 파일을 함께 사용

            if whatif_file is not None:
                wdf = load_event_dataset(whatif_file)
                required_cols = {
                    "date",
                    "item",
                    "sales_qty",
                    "is_weekend",
                    "is_holiday",
                    "is_vacation",
                    "season_period",
                    "precip_type",
                }
                missing_cols = required_cols - set(wdf.columns)

                if missing_cols:
                    st.warning(
                        f"이 기능에는 날씨·행사 정보가 담긴 판매 파일이 필요해요. 빠진 열: {', '.join(sorted(missing_cols))}"
                    )
                elif not ML_AVAILABLE:
                    st.warning("scikit-learn이 설치되어 있지 않아 계산할 수 없어요.")
                else:
                    wi_item = st.selectbox(
                        "품목", sorted(wdf["item"].unique()), key="whatif_item"
                    )

                    item_daily = (
                        wdf[wdf["item"] == wi_item]
                        .groupby("date")
                        .agg(
                            sales_qty=("sales_qty", "sum"),
                            is_weekend=("is_weekend", "first"),
                            is_holiday=("is_holiday", "first"),
                            is_vacation=("is_vacation", "first"),
                            season_period=("season_period", "first"),
                            precip_type=("precip_type", "first"),
                        )
                        .sort_index()
                        .asfreq("D")
                    )
                    item_daily["sales_qty"] = item_daily["sales_qty"].fillna(0.0)
                    item_daily = item_daily.ffill()

                    if len(item_daily) < 30:
                        st.warning("30일 이상의 판매 기록이 있어야 해요.")
                    else:
                        events_full = build_event_dummies(item_daily)
                        beta = fit_event_elasticity(
                            item_daily["sales_qty"], events_full
                        )
                        base_forecast = float(
                            np.clip(
                                predict_holt_trend(item_daily["sales_qty"].values, 1)[
                                    0
                                ],
                                0,
                                None,
                            )
                        )

                        st.markdown("##### 조건 고르기")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            chk_rain = st.checkbox("비", key="wi_rain")
                            chk_snow = st.checkbox("눈", key="wi_snow")
                        with c2:
                            chk_holiday = st.checkbox("공휴일", key="wi_holiday")
                            chk_weekend = st.checkbox("주말", key="wi_weekend")
                        with c3:
                            chk_vacation = st.checkbox("방학", key="wi_vacation")
                        with c4:
                            season_choice = st.selectbox(
                                "시즌",
                                ["없음", "크리스마스", "수능", "밸런타인", "추석"],
                                key="wi_season",
                            )
                        tau_wi = st.slider(
                            "반영 기준",
                            0.0,
                            0.3,
                            0.10,
                            0.01,
                            key="wi_tau",
                            help="조건의 영향이 이 값보다 작으면 무시하고 평소 예측을 그대로 써요. "
                            "0으로 두면 작은 영향도 모두 반영해요.",
                        )

                        def whatif_events(flags=None):
                            """체크박스 상태(+선택적으로 채팅에서 읽은 조건)를 합쳐 이벤트 벡터를 만든다."""
                            f = flags or {k: False for k in WHATIF_KEYWORD_MAP}
                            return {
                                "is_weekend": float(chk_weekend or f["is_weekend"]),
                                "is_holiday": float(chk_holiday or f["is_holiday"]),
                                "is_vacation": float(chk_vacation or f["is_vacation"]),
                                "is_rain": float(chk_rain or f["is_rain"]),
                                "is_snow": float(chk_snow or f["is_snow"]),
                                "is_christmas": float(
                                    season_choice == "크리스마스" or f["is_christmas"]
                                ),
                                "is_suneung": float(
                                    season_choice == "수능" or f["is_suneung"]
                                ),
                                "is_valentine": float(
                                    season_choice == "밸런타인" or f["is_valentine"]
                                ),
                                "is_chuseok": float(
                                    season_choice == "추석" or f["is_chuseok"]
                                ),
                            }

                        # ---------- 즉시 결과 ----------
                        # 체크박스를 건드리는 순간 바로 보이도록, 채팅과 무관하게 항상 계산해 띄운다.
                        live_events = whatif_events()
                        live = whatif_compute(beta, base_forecast, live_events, tau_wi)
                        live_active = [
                            SHAP_LABELS.get(k, k) for k, v in live_events.items() if v
                        ]

                        st.markdown("---")
                        r1, r2, r3 = st.columns(3)
                        r1.metric("평소 예상", f"{base_forecast:.0f}개")
                        r2.metric(
                            "조건 반영 예상",
                            f"{live['final']:.0f}개",
                            delta=f"{live['pct']:+.1f}%",
                        )
                        r3.metric(
                            "권장 준비량",
                            f"{live['production']}개",
                            help="예상보다 10% 넉넉하게 잡은 수예요",
                        )

                        if not live_active:
                            st.caption("조건을 고르면 숫자가 바뀌어요.")
                        else:
                            st.caption(f"고른 조건: {', '.join(live_active)}")

                            # 어떤 이벤트가 얼마나 밀어올리고 끌어내렸는지 보여준다.
                            # 이게 없으면 "비를 켰는데 숫자가 그대로"인 이유를 알 수 없다.
                            contrib_rows = [
                                {
                                    "조건": SHAP_LABELS.get(k, k),
                                    "영향": f"{v * 100:+.1f}%",
                                }
                                for k, v in sorted(
                                    live["contribs"].items(), key=lambda kv: -abs(kv[1])
                                )
                            ]
                            cc1, cc2 = st.columns([1, 1.3])
                            with cc1:
                                st.dataframe(
                                    pd.DataFrame(contrib_rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                            with cc2:
                                if live["gate_on"]:
                                    st.success(
                                        f"조건 영향({live['score']:+.2f})이 반영 기준({tau_wi:.2f})보다 커서 예측에 반영했어요."
                                    )
                                else:
                                    st.warning(
                                        f"조건 영향({live['score']:+.2f})이 반영 기준({tau_wi:.2f})보다 작아서 평소 예상과 같아요. "
                                        f"반영해 보려면 기준을 {max(abs(live['score']) - 0.01, 0.0):.2f} 이하로 낮춰 보세요."
                                    )

                        # ---------- 채팅 ----------
                        st.markdown("---")
                        st.markdown("##### 물어보기")

                        if "whatif_messages" not in st.session_state:
                            st.session_state.whatif_messages = []

                        if st.button("대화 지우기", key="whatif_clear"):
                            st.session_state.whatif_messages = []
                            st.rerun()

                        for msg in st.session_state.whatif_messages:
                            st.chat_message(msg["role"]).write(msg["content"])

                        user_text = st.chat_input(
                            f"예: 내일 비 오고 방학이면 {wi_item} 몇 개 만들까?"
                        )
                        if user_text:
                            st.session_state.whatif_messages.append(
                                {"role": "user", "content": user_text}
                            )
                            text_flags = parse_events_llm(user_text)
                            if text_flags is None:
                                text_flags = parse_whatif_keywords(user_text)

                            events = whatif_events(text_flags)
                            r = whatif_compute(beta, base_forecast, events, tau_wi)
                            active = [
                                SHAP_LABELS.get(k, k) for k, v in events.items() if v
                            ]
                            cond_str = (
                                ", ".join(active) if active else "특별한 이벤트 없음"
                            )
                            reply = whatif_reply_text(
                                wi_item, cond_str, base_forecast, r, tau_wi
                            )

                            ctx = {
                                "item": wi_item,
                                "cond_str": cond_str,
                                "base": base_forecast,
                                "final": r["final"],
                                "pct": r["pct"],
                                "production": r["production"],
                                "score": r["score"],
                                "tau": tau_wi,
                                "gate": "발동" if r["gate_on"] else "미발동",
                            }

                            with st.chat_message("user"):
                                st.markdown(user_text)

                            with st.chat_message("assistant"):
                                try:
                                    streamed = st.write_stream(stream_reply(ctx))
                                except Exception as e:
                                    print(f"[whatif_llm] 스트리밍 실패: {e}")
                                    streamed = None

                                if not streamed:
                                    st.markdown(reply)
                                    streamed = reply

                            st.session_state.whatif_messages.append(
                                {"role": "assistant", "content": streamed}
                            )
            else:
                st.info("날씨·행사 정보가 담긴 판매 CSV를 올리면 쓸 수 있어요.")

        # 탭 6: 이벤트 우선 게이팅(H1) 검증 — 개발자 전용, 맨 뒤에 배치
        if page == PAGE_DEV:
            st.subheader("🌦 이벤트 우선 게이팅 (Layer 2, H1) — Walk-forward 백테스트")
            st.caption(
                "공휴일·방학·시즌·날씨 이벤트를 반영한 예측이, 이벤트를 무시한 기저 시계열 예측보다 "
                "실제로 더 정확한지 최근 기간을 떼어내 검증합니다 (가설 H1). "
                "미래 이벤트는 알 수 없으므로 '미래 예측'이 아니라 과거 데이터로 백테스트합니다."
            )
            event_file = uploaded_file  # 데이터 관리에서 불러온 파일을 함께 사용

            if event_file is not None:
                edf = load_event_dataset(event_file)
                required_cols = {
                    "date",
                    "item",
                    "sales_qty",
                    "is_weekend",
                    "is_holiday",
                    "is_vacation",
                    "season_period",
                    "precip_type",
                }
                missing_cols = required_cols - set(edf.columns)

                if missing_cols:
                    st.warning(
                        f"이 기능에는 날씨·행사 정보가 담긴 판매 파일이 필요해요. 빠진 열: {', '.join(sorted(missing_cols))}"
                    )
                elif not ML_AVAILABLE:
                    st.warning(
                        "scikit-learn이 설치되어 있지 않아 이벤트 탄력도를 추정할 수 없습니다."
                    )
                else:
                    ev_items = sorted(edf["item"].unique())
                    ev_item = st.selectbox("품목 선택", ev_items, key="event_item")
                    horizon = st.slider(
                        "백테스트 검증 기간(일)", 7, 28, 14, key="event_horizon"
                    )
                    tau = st.slider(
                        "이벤트 게이팅 임계치 τ (이 값을 넘는 이벤트 강도만 예측에 반영)",
                        0.0,
                        0.3,
                        0.10,
                        0.01,
                        key="event_tau",
                    )

                    item_daily = (
                        edf[edf["item"] == ev_item]
                        .groupby("date")
                        .agg(
                            sales_qty=("sales_qty", "sum"),
                            is_weekend=("is_weekend", "first"),
                            is_holiday=("is_holiday", "first"),
                            is_vacation=("is_vacation", "first"),
                            season_period=("season_period", "first"),
                            precip_type=("precip_type", "first"),
                        )
                        .sort_index()
                    )

                    if len(item_daily) < horizon + 30:
                        st.warning(
                            "검증에 필요한 데이터(최소 학습 30일 + 검증기간)가 부족합니다."
                        )
                    else:
                        train = item_daily.iloc[:-horizon]
                        test = item_daily.iloc[-horizon:]

                        events = build_event_dummies(item_daily)
                        beta = fit_event_elasticity(
                            train["sales_qty"], events.loc[train.index]
                        )

                        base_fc = predict_prophet(train["sales_qty"], horizon, freq="D")
                        base_fc = np.clip(base_fc, 0, None)

                        # Layer 2 본선: 이벤트를 Prophet 회귀변수로 기저모델에 직접 투입
                        event_fc = predict_prophet_with_events(
                            train["sales_qty"],
                            events.loc[train.index],
                            events.loc[test.index],
                            freq="D",
                        )
                        event_fc = np.clip(event_fc, 0, None)

                        # 대조군(ablation): 기존 곱셈형 사후 게이팅
                        gated_fc, event_score, gate_on = apply_event_gating(
                            base_fc, beta, events.loc[test.index], tau
                        )

                        actual = test["sales_qty"].values
                        wape_base = compute_wape(actual, base_fc)
                        wape_event = compute_wape(actual, event_fc)
                        wape_gated = compute_wape(actual, gated_fc)

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("기저 예측(Layer 1) WAPE", f"{wape_base:.1f}%")
                        m2.metric(
                            "이벤트 회귀(H1) WAPE",
                            f"{wape_event:.1f}%",
                            delta=f"{wape_event - wape_base:+.1f}%p",
                            delta_color="inverse",
                        )
                        m3.metric(
                            "곱셈형 게이팅(대조군) WAPE",
                            f"{wape_gated:.1f}%",
                            delta=f"{wape_gated - wape_base:+.1f}%p",
                            delta_color="inverse",
                        )
                        m4.metric(
                            "게이팅 발동 일수", f"{int(gate_on.sum())} / {horizon}일"
                        )

                        if wape_event < wape_base:
                            st.success(
                                f"H1 지지: '{ev_item}'은 이벤트를 기저모델 회귀변수로 반영했을 때 "
                                f"WAPE가 {wape_base - wape_event:.1f}%p 개선됐습니다."
                            )
                        else:
                            st.warning(
                                f"H1 미지지: '{ev_item}'은 이번 검증 기간에서 이벤트 반영이 오히려 "
                                f"WAPE를 {wape_event - wape_base:.1f}%p 악화시켰습니다."
                            )

                        st.caption(
                            "※ '곱셈형 게이팅'은 기존 융합식 ŷ_base×(1+Σβₖeₖ)를 그대로 둔 **대조군(ablation)**입니다. "
                            "Prophet이 주간 계절성으로 이미 학습한 주말 효과를 β_weekend(+0.35~+0.43)로 한 번 더 "
                            "곱해 이벤트를 이중 계상하기 때문에, 기저보다 나빠지는 것이 정상입니다. "
                            "본선은 이중계상이 구조적으로 없는 '이벤트 회귀' 쪽입니다."
                        )

                        fig3 = go.Figure()
                        fig3.add_trace(
                            go.Scatter(
                                x=test.index,
                                y=actual,
                                name="실제",
                                line=dict(color=BF_NAVY, width=3),
                            )
                        )
                        fig3.add_trace(
                            go.Scatter(
                                x=test.index,
                                y=base_fc,
                                name="기저 예측(Layer 1)",
                                line=dict(color=BF_SKY, dash="dot"),
                            )
                        )
                        fig3.add_trace(
                            go.Scatter(
                                x=test.index,
                                y=event_fc,
                                name="이벤트 회귀 예측(H1)",
                                line=dict(color=BF_BLUE, width=3),
                            )
                        )
                        fig3.add_trace(
                            go.Scatter(
                                x=test.index,
                                y=gated_fc,
                                name="곱셈형 게이팅(대조군)",
                                line=dict(color=BF_GRAY, dash="dash", width=1.5),
                            )
                        )
                        for d, on in zip(test.index, gate_on):
                            if on:
                                fig3.add_vrect(
                                    x0=d - pd.Timedelta(hours=12),
                                    x1=d + pd.Timedelta(hours=12),
                                    fillcolor="#F59E0B",
                                    opacity=0.12,
                                    line_width=0,
                                )
                        fig3.update_layout(
                            height=450,
                            hovermode="x unified",
                            title=f"{ev_item} — 이벤트 게이팅 백테스트 (주황 음영 = 게이팅 발동일)",
                        )
                        st.plotly_chart(fig3, use_container_width=True)

                        with st.expander("📊 추정된 이벤트 탄력도(β) 보기"):
                            beta_df = pd.DataFrame(
                                {
                                    "이벤트": list(beta.keys()),
                                    "탄력도(β)": list(beta.values()),
                                }
                            )
                            beta_df = beta_df.reindex(
                                beta_df["탄력도(β)"]
                                .abs()
                                .sort_values(ascending=False)
                                .index
                            )
                            st.dataframe(
                                beta_df, use_container_width=True, hide_index=True
                            )
                            st.caption(
                                "β > 0: 해당 이벤트일 때 판매량 증가 경향 / β < 0: 감소 경향 "
                                "(학습 구간 기준 선형회귀 추정치, Σβₖ·eₖ가 이벤트 강도 점수)"
                            )
            else:
                st.info(
                    "👈 방학·공휴일·날씨·시즌 이벤트가 포함된 통합 데이터셋 CSV를 업로드하면 H1 가설을 검증할 수 있습니다."
                )
