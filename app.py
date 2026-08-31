import streamlit as st
import pandas as pd
import re
import numpy as np
from datetime import timedelta
from typing import TypedDict, List
import io
import sqlite3
from pathlib import Path
from agent.brief_llm import generate_briefing
from agent.mock_shap import get_contributions, format_contributions
from agent.whatif_llm import parse_events_llm, stream_reply

# --- 팀원 C 파트: 이상탐지(SPC 관리도) · 성능평가 모듈 ---
from detect.control_charts import (
    shewhart_3sigma, ewma_chart, cusum_chart, isolation_forest_flags,
    run_all_detectors, residualize, weekday_residual,
)
# 탐지 결과를 '재고를 어떻게 하라'는 말로 옮기는 계층.
# 통계 용어(3시그마·EWMA·CUSUM)는 화면에 나가지 않는다.
from detect.alerts import build_alerts, build_context, trend_summary, weekday_guide

# --- 계정 인증 (소상공인별 로그인) ---
from auth.ui import (
    require_login, render_sidebar_account, render_account_page, ACCOUNT_VIEW_KEY,
)
from auth import store as auth_store

# 시각화 라이브러리
import plotly.graph_objects as go
import plotly.express as px

# --- 통계/시계열 라이브러리 ---
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests

# --- 머신러닝 라이브러리 (설치 여부 확인) ---
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression
    import xgboost as xgb

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# --- 딥러닝 라이브러리 (설치 여부 확인) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- Prophet 라이브러리 (Layer 1 기저모델, 설치 여부 확인) ---
try:
    import logging as _logging
    from prophet import Prophet

    _logging.getLogger('cmdstanpy').setLevel(_logging.WARNING)
    _logging.getLogger('prophet').setLevel(_logging.WARNING)

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# --- TFT(Temporal Fusion Transformer) 라이브러리 (Layer 1 고도화, 논문용 성능 비교, 설치 여부 확인) ---
try:
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    _logging.getLogger('pytorch_lightning').setLevel(_logging.WARNING)
    _logging.getLogger('lightning').setLevel(_logging.WARNING)

    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

# --- LangGraph 에이전트 라이브러리 (설치 여부 확인) ---
try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# --- SHAP 라이브러리 (설치 여부 확인) ---
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Biz Forecaster — 소상공인 AI 수요예측", layout="wide")

# -----------------------------------------------------------------------------
# 1-1. 로그인 게이트
# -----------------------------------------------------------------------------
# 소상공인마다 자기 매장 데이터만 보도록, 분석 화면 전체를 로그인 뒤로 숨긴다.
# 미로그인 상태에서는 require_login()이 인증 화면을 그리고 None을 돌려주며,
# st.stop()으로 아래 코드를 아예 실행하지 않는다 (데이터가 그려질 여지 자체를 없앤다).
# 동봉된 예시 데이터셋. 데이터가 없어도 화면을 둘러볼 수 있게 한다.
SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "integrated_dataset.csv"

CURRENT_USER = require_login()
if CURRENT_USER is None:
    st.stop()

render_sidebar_account()

# 계정 설정 화면이 열려 있으면 분석 화면 대신 그것만 그린다.
if st.session_state.get(ACCOUNT_VIEW_KEY):
    render_account_page()
    st.stop()


@st.cache_data
def load_data(uploaded_file):
    # 표준 통합 데이터셋(CSV) 로더: date/item/sales_qty를 Date/Item/Quantity로 매핑
    df = pd.read_csv(uploaded_file, parse_dates=['date'])

    # 같은 날짜·품목의 여러 시간대 행을 일 단위로 합산
    daily = (
        df.groupby(['date', 'item'], as_index=False)['sales_qty']
        .sum()
        .rename(columns={'date': 'Date', 'item': 'Item', 'sales_qty': 'Quantity'})
    )
    daily['Quantity'] = daily['Quantity'].astype(float)
    return daily[['Item', 'Date', 'Quantity']]

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()


# -----------------------------------------------------------------------------
# 2. 품목 등급 계산 함수
# -----------------------------------------------------------------------------
def calculate_item_tiers(df):
    # 품목별 총 판매량 집계
    item_stats = df.groupby('Item')['Quantity'].sum().reset_index()
    item_stats = item_stats.sort_values('Quantity', ascending=False)

    # 순위 백분율 계산
    item_stats['Rank_Pct'] = item_stats['Quantity'].rank(pct=True, ascending=False)

    # 등급 부여
    def get_tier(pct):
        if pct <= 0.1:
            return '💎 시그니처 메뉴 (상위 10%)'
        elif pct <= 0.3:
            return '🥇 인기 메뉴 (상위 30%)'
        elif pct <= 0.6:
            return '🥈 스테디셀러 (상위 60%)'
        else:
            return '🥉 일반 품목'

    item_stats['Tier'] = item_stats['Rank_Pct'].apply(get_tier)
    return item_stats[['Item', 'Tier', 'Quantity']]


# -----------------------------------------------------------------------------
# 3. 예측 모델 함수 (추세 반영 강화)
# -----------------------------------------------------------------------------

def predict_linear_trend_force(series, weeks=5):
    try:
        lookback = 8
        recent_data = series[-lookback:] if len(series) >= lookback else series
        n = len(recent_data)
        if n < 2: return [series.mean()] * weeks

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
                series, trend='add', seasonal=None, damped_trend=False
            ).fit(optimized=True)
            # statsmodels 버전에 따라 forecast()가 ndarray 또는 Series를 반환하므로
            # np.asarray로 통일한다 (.values는 ndarray에 없어 AttributeError 발생 가능).
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
        X.append(s_list[i: i + window_size])
        y.append(s_list[i + window_size])
    return np.array(X), np.array(y)


def predict_ml_recursive(model, last_window, weeks=5):
    preds = []
    curr = list(last_window)
    for _ in range(weeks):
        in_row = np.array(curr[-len(last_window):]).reshape(1, -1)
        pred = model.predict(in_row)[0]
        preds.append(pred)
        curr.append(pred)
    return preds


def predict_rf(series, weeks=5):
    if not ML_AVAILABLE: return [0] * weeks
    win = 4
    if len(series) < win + 2: return predict_linear_trend_force(series, weeks)
    X, y = create_ml_dataset(series, win)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return predict_ml_recursive(model, series[-win:], weeks)


def predict_xgboost(series, weeks=5):
    if not ML_AVAILABLE: return [0] * weeks
    win = 4
    if len(series) < win + 2: return predict_linear_trend_force(series, weeks)
    X, y = create_ml_dataset(series, win)
    model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
    model.fit(X, y)
    return predict_ml_recursive(model, series[-win:], weeks)


def predict_prophet(dated_series, periods, freq='D'):
    """Layer 1 기저모델(Prophet). dated_series는 DatetimeIndex를 가진 pd.Series여야
    실제 달력(연간·주간 계절성)을 학습할 수 있다. 실패·미설치·데이터 부족 시
    predict_linear_trend_force로 폴백한다."""
    values = np.asarray(dated_series.values if hasattr(dated_series, 'values') else dated_series, dtype=float)
    if not PROPHET_AVAILABLE or len(values) < 10 or not hasattr(dated_series, 'index'):
        return predict_linear_trend_force(values, periods)
    try:
        train_df = pd.DataFrame({'ds': dated_series.index, 'y': values})
        span_periods = len(train_df)
        is_weekly = freq.startswith('W')
        model = Prophet(
            weekly_seasonality=(not is_weekly),
            yearly_seasonality=(span_periods >= (104 if is_weekly else 365)),
            daily_seasonality=False,
        )
        model.fit(train_df)
        future = model.make_future_dataframe(periods=periods, freq=freq, include_history=False)
        forecast = model.predict(future)
        return forecast['yhat'].to_numpy()[:periods]
    except Exception:
        return predict_linear_trend_force(values, periods)



def predict_prophet_with_events(train_series, train_events, test_events, freq='D'):
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
    values = np.asarray(train_series.values if hasattr(train_series, 'values') else train_series,
                        dtype=float)
    if not PROPHET_AVAILABLE or len(values) < 10 or not hasattr(train_series, 'index'):
        return predict_prophet(train_series, periods, freq=freq)
    try:
        cols = list(train_events.columns)
        is_weekly = freq.startswith('W')
        model = Prophet(
            weekly_seasonality=(not is_weekly),
            yearly_seasonality=(len(values) >= (104 if is_weekly else 365)),
            daily_seasonality=False,
        )
        for c in cols:
            model.add_regressor(c)

        train_df = pd.DataFrame({'ds': train_series.index, 'y': values})
        for c in cols:
            train_df[c] = np.asarray(train_events[c].values, dtype=float)
        model.fit(train_df)

        future = pd.DataFrame({'ds': test_events.index})
        for c in cols:
            future[c] = np.asarray(test_events[c].values, dtype=float)
        return model.predict(future)['yhat'].to_numpy()[:periods]
    except Exception:
        return predict_prophet(train_series, periods, freq=freq)


def predict_tft(dated_series, periods, freq='D'):
    """Layer 1 고도화 모델(TFT, Temporal Fusion Transformer) — 논문용 Prophet 대비
    성능 비교 목적. 단일 품목 시계열에 대해 짧게 학습(few-epoch)한 뒤,
    **관측 구간 이후의 미래 periods 구간**을 예측한다. 데이터가 부족
    (encoder+prediction 구간 확보 불가)하거나 미설치·학습 실패 시
    predict_prophet으로 폴백한다."""
    values = np.asarray(dated_series.values if hasattr(dated_series, 'values') else dated_series, dtype=float)
    n = len(values)
    encoder_len = min(30, max(10, n - periods - 5))
    if not TFT_AVAILABLE or n < encoder_len + periods + 5:
        return predict_prophet(dated_series, periods, freq=freq)
    try:
        torch.manual_seed(42)
        df = pd.DataFrame({
            'time_idx': np.arange(n),
            'group': 'series',
            'value': values,
        })

        # 학습은 관측된 전체 이력(0 ~ n-1)을 사용한다. 예측 대상이 '미래'이므로
        # 가장 최근 구간을 학습에서 떼어놓을 이유가 없다.
        training = TimeSeriesDataSet(
            df,
            time_idx='time_idx',
            target='value',
            group_ids=['group'],
            min_encoder_length=max(1, encoder_len // 2),
            max_encoder_length=encoder_len,
            min_prediction_length=1,
            max_prediction_length=periods,
            time_varying_unknown_reals=['value'],
            target_normalizer=GroupNormalizer(groups=['group']),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        train_dataloader = training.to_dataloader(train=True, batch_size=16, num_workers=0)

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=8,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(),
            optimizer='adam',
            log_interval=-1,
        )

        trainer = pl.Trainer(
            max_epochs=8,
            accelerator='cpu',
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(tft, train_dataloaders=train_dataloader)

        # --- 미래 구간 예측 ---
        # TimeSeriesDataSet(predict=True)는 '넘겨준 데이터프레임의 마지막 예측창'을 고른다.
        # 관측 데이터만 넘기면 디코더가 과거 마지막 periods 구간을 가리키므로(=예측이 아니라 재현),
        # 인코더(관측 마지막 encoder_len) + 디코더(아직 관측되지 않은 미래 periods) 프레임을
        # 직접 만들어 넘긴다. 미래 행의 'value'는 타깃이라 모델 입력으로 쓰이지 않으며,
        # 데이터셋 구성을 위한 자리표시자로 마지막 관측값을 복사해 둔다.
        last_idx = int(df['time_idx'].iloc[-1])
        encoder_df = df[df['time_idx'] > last_idx - encoder_len]
        future_df = pd.concat(
            [df.iloc[[-1]].assign(time_idx=last_idx + i) for i in range(1, periods + 1)],
            ignore_index=True,
        )
        predict_df = pd.concat([encoder_df, future_df], ignore_index=True)

        predict_ds = TimeSeriesDataSet.from_dataset(
            training, predict_df, predict=True, stop_randomization=True
        )
        predict_dl = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

        raw_predictions = tft.predict(predict_dl, mode='prediction')
        forecast = np.asarray(raw_predictions[0]).flatten()[:periods]
        if len(forecast) < periods:
            return predict_prophet(dated_series, periods, freq=freq)
        return forecast
    except Exception:
        return predict_prophet(dated_series, periods, freq=freq)


def predict_lstm(series, weeks=5):
    if not TF_AVAILABLE: return [0] * weeks
    win = 4
    if len(series) < win + 5: return predict_linear_trend_force(series, weeks)
    scaler = MinMaxScaler(feature_range=(0, 1))
    s_scaled = scaler.fit_transform(np.array(series).reshape(-1, 1))
    X, y = [], []
    for i in range(len(s_scaled) - win):
        X.append(s_scaled[i: i + win])
        y.append(s_scaled[i + win])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(win, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0, batch_size=2)

    preds = []
    curr = s_scaled[-win:]
    for _ in range(weeks):
        pred_sc = model.predict(curr.reshape(1, win, 1), verbose=0)[0][0]
        preds.append(pred_sc)
        curr = np.append(curr[1:], [[pred_sc]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# -----------------------------------------------------------------------------
# 4. LangGraph 에이전트 (조회 → 예측 → 해석 → 브리핑)
# -----------------------------------------------------------------------------
# ※ Layer 2(이벤트 게이팅)·Layer 3(품목간 수요연쇄)·SHAP 기여도 분해는 아직 로드맵 개발 중이라
#   이 에이전트의 '해석' 단계는 현재 계산된 예측치의 통계(최근 평균 대비 변화율)만 근거로 사용한다.
#   이벤트·품목연쇄 파이프라인이 붙으면 _interpret_node 안에서 해당 기여도를 추가로 반영하면 된다.

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
        "shap_top": format_contributions(get_contributions(state["item"])),
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
        f"{trend_phrase} 특히 {state['peak_week']} 주간에 가장 높은 수요가 예상됩니다. {action_phrase}\n\n"
        "※ 본 브리핑은 시계열 기저 예측(Layer 1) 통계 기준이며, 이벤트 우선 게이팅(H1)·품목간 수요연쇄(H2′)·"
        "SHAP 기여도 분해는 로드맵 개발 중으로 아직 반영되지 않았습니다."
    )
    llm_text=generate_briefing(state)
    if llm_text:
        briefing=llm_text
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


# -----------------------------------------------------------------------------
# 5. 이벤트 우선 게이팅 (Layer 2, H1) — 통합 이벤트 데이터셋 기반 Walk-forward 백테스트
# -----------------------------------------------------------------------------
# 데이터셋의 미래(예: 다음 5주)의 실제 공휴일·날씨는 알 수 없으므로, '미래 예측'이 아니라
# 최근 N일을 떼어내 실제값과 비교하는 백테스트로 H1(이벤트 우선 예측이 기저 예측보다 정확한가)을 검증한다.

@st.cache_data
def load_event_dataset(uploaded_file):
    return pd.read_csv(uploaded_file, parse_dates=['date'])


def build_event_dummies(daily):
    """일별 이벤트 플래그(요일·공휴일·방학·시즌·날씨)를 0/1 더미로 변환."""
    e = pd.DataFrame(index=daily.index)
    e['is_weekend'] = daily['is_weekend'].astype(float)
    e['is_holiday'] = daily['is_holiday'].astype(float)
    e['is_vacation'] = daily['is_vacation'].astype(float)
    e['is_rain'] = (daily['precip_type'] == '비').astype(float)
    e['is_snow'] = (daily['precip_type'] == '눈').astype(float)
    e['is_christmas'] = (daily['season_period'] == '크리스마스시즌').astype(float)
    e['is_suneung'] = (daily['season_period'] == '수능시즌').astype(float)
    e['is_valentine'] = (daily['season_period'] == '밸런타인시즌').astype(float)
    e['is_chuseok'] = (daily['season_period'] == '추석시즌').astype(float)
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


# -----------------------------------------------------------------------------
# 7. 실시간 이상탐지 — SPC 관리도 3종 + Isolation Forest
# -----------------------------------------------------------------------------
# 구현체는 detect/ 패키지로 분리했다(detect/control_charts.py, detect/evaluation.py).
# app.py는 화면만 담당하고, 탐지 로직은 UI 없이 단독으로 테스트할 수 있게 떼어낸 것이다.
#
# 팀원 C 파트에서 이번에 보강한 것
#   - CUSUM 관리도 추가 (기존에는 3시그마·EWMA뿐이라 점진적 드리프트를 놓치고 있었다)
#   - 요일 계절성을 제거한 잔차 관리도 옵션 (SPC의 정상상태 가정을 맞춰준다)
#   - 오탐/미탐 트레이드오프 정량화 (detect/evaluation.py, 탭 8)
#
# 아래 세 함수는 기존 호출부와의 호환을 위해 남겨둔 얇은 래퍼다.

def rolling_3sigma_flags(series, window=28, min_periods=10, L=3.0, non_negative=True):
    """3시그마(Shewhart) 관리도 — detect.control_charts.shewhart_3sigma 래퍼."""
    return shewhart_3sigma(series, L=L, window=window,
                            min_periods=min_periods, non_negative=non_negative)


def ewma_flags(series, lam=0.2, L=3, window=28, min_periods=10, non_negative=True):
    """EWMA 관리도 — detect.control_charts.ewma_chart 래퍼."""
    return ewma_chart(series, lam=lam, L=L, window=window,
                       min_periods=min_periods, non_negative=non_negative)


def cusum_flags(series, k=0.5, h=5.0, window=28, min_periods=10):
    """CUSUM 관리도 — detect.control_charts.cusum_chart 래퍼 (이번에 새로 추가)."""
    return cusum_chart(series, k=k, h=h, window=window, min_periods=min_periods)


# -----------------------------------------------------------------------------
# 8. SHAP 기여도 분석 (XAI)
# -----------------------------------------------------------------------------
# build_event_dummies()로 만든 이벤트 피처에 기온·추세를 더해 RandomForest로 학습하고,
# SHAP TreeExplainer로 각 예측을 '기저치 + 이벤트별 기여도'로 분해한다.
# 여기서 나오는 (피처, 기여도) 쌍은 LangGraph 에이전트(_interpret_node)의 해석 단계에
# 그대로 연결할 수 있도록 설계했다.

SHAP_LABELS = {
    'is_weekend': '주말', 'is_holiday': '공휴일', 'is_vacation': '방학',
    'is_rain': '비', 'is_snow': '눈', 'is_christmas': '크리스마스시즌',
    'is_suneung': '수능시즌', 'is_valentine': '밸런타인시즌', 'is_chuseok': '추석시즌',
    'temperature': '기온', 'trend': '추세',
}


def build_shap_features(daily):
    """이벤트 더미 + 기온 + 추세(day index)로 이루어진 SHAP 학습용 피처 행렬."""
    features = build_event_dummies(daily)
    features['temperature'] = daily['temperature']
    features['trend'] = np.arange(len(daily))
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
    top = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(top_n)
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


# -----------------------------------------------------------------------------
# 10. What-if 채팅 (D팀) — 키워드 기반 시나리오 파싱 + H1 게이팅 모델로 실계산
# -----------------------------------------------------------------------------
# 실제 LLM API 키가 연결돼 있지 않아, 자연어 이해는 키워드 매칭으로 단순화했다.
# 대신 응답에 쓰이는 예측치·권장 생산량은 build_event_dummies/fit_event_elasticity로
# 학습한 실제 H1 모델에서 계산한 값이다 (지어낸 숫자가 아님).

WHATIF_KEYWORD_MAP = {
    'is_rain': ['비', '우천', '강수'],
    'is_snow': ['눈', '폭설'],
    'is_holiday': ['공휴일', '연휴', '휴일'],
    'is_vacation': ['방학'],
    'is_weekend': ['주말', '토요일', '일요일'],
    'is_christmas': ['크리스마스', '성탄'],
    'is_suneung': ['수능'],
    'is_valentine': ['밸런타인', '발렌타인'],
    'is_chuseok': ['추석'],
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
    final = float(np.clip(base_forecast * (1 + score) if gate_on else base_forecast, 0, None))
    return {
        'score': score,
        'gate_on': gate_on,
        'final': final,
        'production': int(np.ceil(final * 1.1)),
        'pct': (final - base_forecast) / base_forecast * 100 if base_forecast > 0 else 0.0,
        'contribs': contribs,
    }


def whatif_reply_text(item, cond_str, base_forecast, r, tau):
    """계산 결과를 한 문단 답변으로 만든다 (LLM 미연동 시 사용하는 기본 답변)."""
    reply = (
        f"조건({cond_str})을 반영하면 '{item}' 예상 판매량은 "
        f"{base_forecast:.0f}개 → **{r['final']:.0f}개** ({r['pct']:+.1f}%)입니다. "
        f"안전재고 10%를 더한 권장 생산량은 **{r['production']}개**입니다."
    )
    if not r['gate_on']:
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


# -----------------------------------------------------------------------------
# 3. 메인 UI 구성
# -----------------------------------------------------------------------------
st.title(f"🥐 {CURRENT_USER['shop_name']} — AI 수요 예측")
st.write("판매 활성도(기간)와 판매 규모(베스트셀러 등급) 필터를 활용하여 원하는 품목을 쉽게 찾으세요.")

# --- 사용자별 데이터 업로드 ---
# 업로드한 파일은 계정별로 분리된 data/users/<user_id>/ 아래에만 저장된다.
# 다음 접속 때 다시 올릴 필요 없이 이전 파일을 그대로 불러올 수 있다.
with st.sidebar:
    st.header("📂 1. 파일 업로드")
    uploaded_file = st.file_uploader("판매내역 CSV 파일을 선택하세요", type=['csv'])

    if uploaded_file is not None:
        try:
            saved_path = auth_store.user_data_dir(CURRENT_USER['id']) / uploaded_file.name
            raw_bytes = uploaded_file.getvalue()
            saved_path.write_bytes(raw_bytes)
            auth_store.record_dataset(
                CURRENT_USER['id'], uploaded_file.name, saved_path,
                n_rows=max(len(raw_bytes.splitlines()) - 1, 0),
            )
            uploaded_file.seek(0)   # 저장하며 소비한 스트림을 되감아야 아래 로더가 읽는다
        except Exception as e:
            st.warning(f"파일 보관에 실패했습니다(분석은 그대로 진행됩니다): {e}")

    if uploaded_file is None and SAMPLE_DATA_PATH.exists():
        if st.checkbox("동봉된 예시 데이터로 둘러보기", key='main_sample'):
            uploaded_file = str(SAMPLE_DATA_PATH)

    my_datasets = auth_store.list_datasets(CURRENT_USER['id'])
    if uploaded_file is None and my_datasets:
        st.caption("이전에 올린 파일 다시 사용하기")
        options = {
            f"{d['filename']}  ({d['uploaded_at'][:16]})": d['stored_path']
            for d in my_datasets
        }
        picked = st.selectbox("내 데이터", ["(선택 안 함)"] + list(options.keys()),
                               label_visibility='collapsed')
        if picked != "(선택 안 함)":
            chosen = Path(options[picked])
            if chosen.exists():
                uploaded_file = str(chosen)
            else:
                st.warning("저장된 파일을 찾을 수 없습니다. 다시 업로드해주세요.")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df.empty:
        st.error("데이터 로드 실패")
    else:
        max_date = df['Date'].max()

        # [전처리] 품목 등급 산출
        tier_df = calculate_item_tiers(df)
        # 등급 정보를 메인 데이터프레임과 병합할 수도 있지만, 필터링용 리스트로 활용

        st.sidebar.success(f"로드 완료! ({len(df):,}건)")

        tab_detail, tab_top, tab_alert, tab_shap, tab_whatif, tab_h1 = st.tabs([
            "📊 품목별 상세 분석 (Ensemble)", "🏆 베스트셀러 TOP N",
            "📦 판매 이상 알림", "🧮 SHAP 기여도", "💬 What-if 채팅",
            "🌦 이벤트 우선 예측 (H1)"
        ])

        # 탭 1: 품목별 상세 분석
        with tab_detail:
            st.subheader("품목 정밀 분석 및 필터링")

            # --- [필터 섹션] ---
            with st.expander("🔎 품목 필터 옵션 (클릭하여 펼치기)", expanded=True):
                f_col1, f_col2 = st.columns(2)

                with f_col1:
                    st.markdown("##### 1. 판매 활성도 기준 (Recency)")
                    activity_opt = st.radio(
                        "최근 판매일 기준:",
                        ('전체', '최근 3개월', '최근 6개월', '최근 1년'),
                        index=0,
                        horizontal=True
                    )

                with f_col2:
                    st.markdown("##### 2. 판매 규모 등급 (Volume)")
                    # 등급 목록 생성 (순서 보장)
                    tier_options = ['전체', '💎 시그니처 메뉴 (상위 10%)', '🥇 인기 메뉴 (상위 30%)', '🥈 스테디셀러 (상위 60%)', '🥉 일반 품목']
                    tier_opt = st.selectbox("품목 등급 기준:", tier_options, index=0)

            # --- [필터링 로직 적용] ---
            # 1. 기간 필터링
            last_dates = df.groupby('Item')['Date'].max().reset_index()
            if activity_opt == '최근 3개월':
                cutoff = max_date - timedelta(days=90)
                active_item_list = last_dates[last_dates['Date'] >= cutoff]['Item'].tolist()
            elif activity_opt == '최근 6개월':
                cutoff = max_date - timedelta(days=180)
                active_item_list = last_dates[last_dates['Date'] >= cutoff]['Item'].tolist()
            elif activity_opt == '최근 1년':
                cutoff = max_date - timedelta(days=365)
                active_item_list = last_dates[last_dates['Date'] >= cutoff]['Item'].tolist()
            else:
                active_item_list = last_dates['Item'].tolist()

            # 2. 등급 필터링
            if tier_opt != '전체':
                tier_item_list = tier_df[tier_df['Tier'] == tier_opt]['Item'].tolist()
            else:
                tier_item_list = tier_df['Item'].tolist()

            # 교집합 (두 조건 모두 만족)
            final_item_list = list(set(active_item_list) & set(tier_item_list))
            final_item_list.sort()

            # --- [선택 UI] ---
            st.divider()
            c_info, c_select = st.columns([1, 2])

            with c_info:
                st.metric("조건 만족 품목 수", f"{len(final_item_list)} 개")
                if tier_opt != '전체':
                    st.caption(f"선택 등급: {tier_opt}")
                if activity_opt != '전체':
                    st.caption(f"판매 기간: {activity_opt} 이내")

            with c_select:
                search_txt = st.text_input("품목명 검색", placeholder="예: 식빵")
                display_list = [c for c in final_item_list if search_txt in c] if search_txt else final_item_list
                selected_item = st.selectbox("분석 대상 선택:", display_list, index=None, placeholder="목록에서 선택하세요...")

            # --- [분석 실행] ---
            if selected_item:
                st.markdown(f"### 🎯 '{selected_item}' 앙상블 예측")

                # 품목 등급 표시
                item_tier_info = tier_df[tier_df['Item'] == selected_item]['Tier'].values[0]
                st.info(f"이 품목은 **{item_tier_info}** 입니다.")

                item_df = df[df['Item'] == selected_item].sort_values('Date')
                item_weekly = item_df.set_index('Date').resample('W-MON')['Quantity'].sum()

                start_date = item_weekly[item_weekly > 0].index.min()
                if pd.isna(start_date): start_date = item_weekly.index.min()

                full_idx = pd.date_range(start=start_date, end=max_date, freq='W-MON')
                item_weekly = item_weekly.reindex(full_idx, fill_value=0)
                series_data = item_weekly.values

                with st.spinner("예측 모델 분석 중..."):
                    p_linear = predict_linear_trend_force(series_data, 5)
                    p_holt = predict_holt_trend(series_data, 5)
                    p_arima = predict_arima_trend(series_data, 5)
                    p_prophet = predict_prophet(item_weekly, 5, freq='W-MON') if PROPHET_AVAILABLE else [0] * 5
                    p_tft = predict_tft(item_weekly, 5, freq='W-MON') if TFT_AVAILABLE else [0] * 5

                    p_rf = predict_rf(series_data, 5) if ML_AVAILABLE else [0] * 5
                    p_xgb = predict_xgboost(series_data, 5) if ML_AVAILABLE else [0] * 5
                    p_lstm = predict_lstm(series_data, 5) if TF_AVAILABLE else [0] * 5

                    valid_preds = [p_linear, p_holt, p_arima]
                    if PROPHET_AVAILABLE: valid_preds.append(p_prophet)
                    if TFT_AVAILABLE: valid_preds.append(p_tft)
                    if ML_AVAILABLE: valid_preds.extend([p_rf, p_xgb])
                    if TF_AVAILABLE: valid_preds.append(p_lstm)

                    ens_pred = np.mean(valid_preds, axis=0)
                    ens_pred = [round(max(0, x), 1) for x in ens_pred]

                dates_str = [(max_date + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, 6)]

                res_df = pd.DataFrame({
                    '날짜': dates_str,
                    '앙상블(최종)': ens_pred,
                    '선형추세': [round(max(0, x), 1) for x in p_linear],
                    'Holt': [round(max(0, x), 1) for x in p_holt],
                    'Prophet': [round(max(0, x), 1) for x in p_prophet],
                    'TFT': [round(max(0, x), 1) for x in p_tft],
                })

                st.subheader("📋 예측 결과표")
                st.dataframe(res_df.set_index('날짜'), use_container_width=True)
                st.download_button("💾 엑셀 다운로드", data=to_excel(res_df), file_name=f"{selected_item}_예측.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.markdown("---")
                st.subheader("📈 추세 그래프")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=item_weekly.index[-20:], y=item_weekly.values[-20:], name='과거 판매실적',
                                         line=dict(color='gray', dash='dot')))
                fig.add_trace(go.Scatter(x=pd.to_datetime(res_df['날짜']), y=res_df['앙상블(최종)'], name='앙상블(추세반영)',
                                         line=dict(color='red', width=4)))
                fig.add_trace(go.Scatter(x=pd.to_datetime(res_df['날짜']), y=res_df['선형추세'], name='선형추세',
                                         line=dict(color='blue', width=1, dash='dot')))
                fig.update_layout(height=500, hovermode="x unified", title=f"{selected_item} 향후 5주 판매 추세")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("🤖 에이전트 브리핑 (조회 → 예측 → 해석 → 브리핑)")
                if LANGGRAPH_AVAILABLE:
                    with st.spinner("에이전트가 브리핑을 생성하는 중..."):
                        agent = build_briefing_agent()
                        agent_state = agent.invoke({
                            "item": selected_item,
                            "tier": item_tier_info,
                            "history": [float(v) for v in item_weekly.values],
                            "forecast": [float(v) for v in ens_pred],
                            "dates": dates_str,
                        })
                    st.chat_message("assistant").write(agent_state["briefing"])
                else:
                    st.warning("langgraph가 설치되어 있지 않습니다. `pip install langgraph` 후 에이전트 브리핑을 사용할 수 있습니다.")

            else:
                # 미선택 시 목록 보여주기
                if display_list:
                    st.markdown(f"### 📋 조회된 품목 목록 ({len(display_list)}개)")

                    # 상세 정보 병합 (마지막 판매일, 총판매량, 등급)
                    list_df = df[df['Item'].isin(display_list)].groupby('Item').agg(
                        마지막판매일=('Date', 'max'),
                        총판매량=('Quantity', 'sum')
                    ).reset_index()

                    # 등급 정보 병합
                    list_df = pd.merge(list_df, tier_df[['Item', 'Tier']], on='Item', how='left')
                    list_df['마지막판매일'] = list_df['마지막판매일'].dt.strftime('%Y-%m-%d')
                    list_df = list_df.sort_values(['총판매량', '마지막판매일'], ascending=[False, False])  # 판매량 순 정렬

                    st.dataframe(list_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("조건에 맞는 품목이 없습니다. 필터를 변경해보세요.")

        # 탭 2: 베스트셀러 TOP N
        with tab_top:
            st.subheader("🏆 베스트셀러 품목 일괄 예측")
            top_n = st.radio("분석 개수:", [10, 20, 30], horizontal=True)

            if st.button("Top N 분석 시작", type="primary"):
                top_items = df.groupby('Item')['Quantity'].sum().nlargest(top_n).index.tolist()
                results = []
                dates_str = [(max_date + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, 6)]

                bar = st.progress(0)
                for idx, item in enumerate(top_items):
                    i_df = df[df['Item'] == item].sort_values('Date')
                    i_weekly = i_df.set_index('Date').resample('W-MON')['Quantity'].sum()
                    start_d = i_weekly[i_weekly > 0].index.min()
                    if pd.isna(start_d): start_d = i_weekly.index.min()
                    i_series = i_weekly.reindex(pd.date_range(start=start_d, end=max_date, freq='W-MON'),
                                                fill_value=0).values

                    p1 = predict_linear_trend_force(i_series, 5)
                    p2 = predict_holt_trend(i_series, 5)
                    avg = np.mean([p1, p2], axis=0)

                    row = {'품목명': item}
                    for i, d in enumerate(dates_str): row[f'{i + 1}주차 ({d})'] = round(max(0, avg[i]), 1)
                    results.append(row)
                    bar.progress((idx + 1) / top_n)

                top_df = pd.DataFrame(results)
                st.dataframe(top_df, use_container_width=True)
                st.download_button("💾 Top N 결과 엑셀 다운로드", data=to_excel(top_df), file_name=f"Top{top_n}_예측.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.markdown("---")
                if st.toggle("종합 그래프 보기", value=True):
                    plot_df = top_df.melt(id_vars=['품목명'], var_name='주차', value_name='예측수량')
                    fig2 = px.line(plot_df, x='주차', y='예측수량', color='품목명', markers=True, title=f"Top {top_n} 품목 판매 추세")
                    fig2.update_layout(height=600, hovermode="x unified")
                    st.plotly_chart(fig2, use_container_width=True)

        # 탭 3: 판매 이상 알림 (재고 판단용)
        with tab_alert:
            st.subheader("📦 판매 이상 알림")
            st.caption(
                "평소와 다르게 팔린 날을 찾아드립니다. "
                "'평소'는 **직전 4주간 같은 요일의 평균**입니다 — 토요일은 토요일끼리 비교합니다."
            )

            alert_file = st.file_uploader(
                "판매 데이터 CSV 업로드 (date, item, sales_qty 컬럼 포함)",
                type=['csv'], key='anomaly_csv'
            )
            if alert_file is None and SAMPLE_DATA_PATH.exists():
                if st.checkbox("동봉된 예시 데이터로 둘러보기", key='anomaly_sample'):
                    alert_file = str(SAMPLE_DATA_PATH)

            if alert_file is None:
                st.info("👆 판매 데이터를 올리시면 재고 판단에 쓸 알림을 만들어 드립니다.")
            else:
                adf = load_event_dataset(alert_file)
                missing_cols = {'date', 'item', 'sales_qty'} - set(adf.columns)

                if missing_cols:
                    st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")
                elif not ML_AVAILABLE:
                    st.warning("분석에 필요한 라이브러리(scikit-learn)가 설치되어 있지 않습니다.")
                else:
                    daily_total = adf.groupby('date')['sales_qty'].sum().asfreq('D').fillna(0.0)
                    all_items = sorted(adf['item'].unique())
                    by_item = {
                        it: adf[adf['item'] == it].groupby('date')['sales_qty'].sum()
                              .reindex(daily_total.index, fill_value=0.0)
                        for it in all_items
                    }

                    if len(daily_total) < 40:
                        st.warning("분석하려면 최소 40일치 데이터가 필요합니다.")
                    else:
                        period = st.radio(
                            "확인할 기간", [30, 60, 90], index=0, horizontal=True,
                            format_func=lambda d: f"최근 {d}일", key='alert_period'
                        )

                        with st.spinner("판매 흐름을 살펴보는 중입니다..."):
                            # 시즌·공휴일·날씨를 알면 '왜 그랬는지'까지 말해줄 수 있다.
                            context = build_context(adf)
                            alerts = build_alerts(by_item, daily_total, all_items,
                                                   recent_days=period, context=context)
                            trends = {it: trend_summary(by_item[it]) for it in all_items}

                        n_high = sum(1 for a in alerts if a['severity'] == '높음')
                        n_short = sum(1 for a in alerts if a['pct'] > 0)
                        n_over = sum(1 for a in alerts if a['pct'] < 0)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("눈여겨볼 날", f"{len(alerts)}건",
                                  help=f"최근 {period}일 중 평소와 20% 이상 차이 난 날")
                        m2.metric("품절 위험 (많이 팔린 날)", f"{n_short}건")
                        m3.metric("재고 과잉 위험 (적게 팔린 날)", f"{n_over}건")

                        # ---------- 1. 알림 카드 ----------
                        st.markdown("### ⚠️ 이런 날이 있었어요")
                        if not alerts:
                            st.success(
                                f"최근 {period}일 동안 평소와 크게 다른 날이 없었습니다. "
                                "판매가 안정적입니다."
                            )
                        else:
                            n_event = sum(1 for a in alerts if a['is_event'])
                            if n_event:
                                st.caption(
                                    f"이 가운데 {n_event}건은 시즌·공휴일·날씨로 설명되는 날입니다. "
                                    "**원인을 알 수 없는 날을 먼저** 보여드립니다."
                                )
                            elif n_high:
                                st.caption(f"특히 눈여겨볼 것 {n_high}건을 맨 위에 두었습니다.")

                            show_items = st.multiselect(
                                "품목으로 좁혀보기 (비워두면 전체)",
                                sorted({a['item'] for a in alerts}), key='alert_filter'
                            )
                            shown = [a for a in alerts if not show_items or a['item'] in show_items]

                            for a in shown[:20]:
                                with st.container(border=True):
                                    if a['is_event']:
                                        tag = '🗓 이유를 아는 날'
                                    else:
                                        tag = {'높음': '🔴 눈여겨보세요',
                                               '보통': '🟡 참고하세요',
                                               '낮음': '⚪ 가벼운 차이'}[a['severity']]
                                    st.markdown(f"**{a['icon']} {a['headline']}**")
                                    st.caption(f"{tag}  ·  {a['detail']}")
                                    st.markdown(f"→ {a['meaning']} {a['action']}")
                                    if a['cause']:
                                        st.caption(a['cause'])

                            if len(shown) > 20:
                                st.caption(f"이 밖에 {len(shown) - 20}건이 더 있습니다.")

                        # ---------- 2. 요즘 추세 ----------
                        st.markdown("### 📈 요즘 잘 나가고 있나요?")
                        st.caption("하루 튄 것보다 **몇 주에 걸친 변화**가 발주량 조정에 더 중요합니다.")

                        trend_rows = []
                        for it, t in trends.items():
                            if t is None:
                                continue
                            trend_rows.append({
                                '품목': it,
                                '방향': ('▲ 늘고 있음' if t['direction'] == '상승' else '▼ 줄고 있음'),
                                '최근 4주 변화': f"{t['change'] * 100:+.0f}%",
                                '이렇게 해보세요': t['action'],
                            })

                        if trend_rows:
                            trend_df = pd.DataFrame(trend_rows).sort_values(
                                '최근 4주 변화',
                                key=lambda s: s.str.rstrip('%').astype(float),
                                ascending=False,
                            )
                            st.dataframe(trend_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("최근 4주간 뚜렷하게 늘거나 줄어든 품목이 없습니다.")

                        # ---------- 3. 요일별 준비량 ----------
                        st.markdown("### 📅 요일별 준비량 가이드")
                        st.caption("최근 4주 실적을 요일별로 모은 표입니다. 내일 몇 개 준비할지 정할 때 쓰세요.")

                        guide_item = st.selectbox("품목 선택", all_items, key='guide_item')
                        guide = weekday_guide(by_item[guide_item])

                        if guide is None:
                            st.info("이 품목은 데이터가 부족해 요일별 가이드를 만들 수 없습니다.")
                        else:
                            gc1, gc2 = st.columns([1, 1.4])
                            with gc1:
                                st.dataframe(guide, use_container_width=True, hide_index=True)
                                st.caption(
                                    "**권장 준비량**은 평균보다 10% 여유를 둔 값입니다. "
                                    "품절보다 약간의 여유가 낫다는 기준이며, 폐기가 부담되면 평균에 맞추세요."
                                )
                            with gc2:
                                # 최근 60일 판매 흐름 + 평소 범위
                                s = by_item[guide_item].tail(60)
                                _, exp = weekday_residual(by_item[guide_item])
                                exp = exp.reindex(s.index)

                                figg = go.Figure()
                                figg.add_trace(go.Scatter(
                                    x=s.index, y=exp * 1.3, name='평소 범위(위)',
                                    line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                figg.add_trace(go.Scatter(
                                    x=s.index, y=exp * 0.7, name='평소 범위',
                                    line=dict(width=0), fill='tonexty',
                                    fillcolor='rgba(120,170,220,0.18)', hoverinfo='skip'))
                                figg.add_trace(go.Scatter(
                                    x=s.index, y=s, name='실제 판매량',
                                    line=dict(color='#2C3E50', width=2)))

                                hit_dates = [a['date'] for a in alerts if a['item'] == guide_item
                                             and a['date'] in s.index]
                                if hit_dates:
                                    figg.add_trace(go.Scatter(
                                        x=hit_dates, y=s.loc[hit_dates], mode='markers',
                                        name='평소와 달랐던 날',
                                        marker=dict(color='#E74C3C', size=10, symbol='circle')))

                                figg.update_layout(
                                    height=330, hovermode='x unified',
                                    title=f"{guide_item} — 최근 60일",
                                    margin=dict(t=45, b=30, l=10, r=10),
                                    legend=dict(orientation='h', yanchor='bottom', y=1.0),
                                )
                                st.plotly_chart(figg, use_container_width=True)

                        # ---------- 설명 ----------
                        with st.expander("❓ 이 알림은 어떻게 만들어지나요?"):
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
        with tab_shap:
            st.subheader("🧮 SHAP 기여도 분석 (XAI)")
            st.caption(
                "예측치를 '기저치 + 이벤트별 기여도'로 분해합니다. 각 예측이 왜 그 값이 나왔는지 "
                "품목·날짜 단위로 근거를 확인할 수 있습니다."
            )
            shap_file = st.file_uploader(
                "이벤트 통합 데이터셋 CSV 업로드 (date, item, sales_qty, 이벤트·기온 컬럼 포함)",
                type=['csv'], key='shap_csv'
            )

            if shap_file is None and SAMPLE_DATA_PATH.exists():

                if st.checkbox('동봉된 예시 데이터로 둘러보기', key='shap_sample'):

                    shap_file = str(SAMPLE_DATA_PATH)


            if shap_file is not None:
                sdf = load_event_dataset(shap_file)
                required_cols = {'date', 'item', 'sales_qty', 'is_weekend', 'is_holiday',
                                  'is_vacation', 'season_period', 'precip_type', 'temperature'}
                missing_cols = required_cols - set(sdf.columns)

                if missing_cols:
                    st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")
                elif not ML_AVAILABLE:
                    st.warning("scikit-learn이 설치되어 있지 않아 모델을 학습할 수 없습니다.")
                elif not SHAP_AVAILABLE:
                    st.warning("shap이 설치되어 있지 않습니다. `pip install shap` 후 이용할 수 있습니다.")
                else:
                    shap_items = sorted(sdf['item'].unique())
                    shap_item = st.selectbox("품목 선택", shap_items, key='shap_item')
                    shap_horizon = st.slider("SHAP 분석 대상 기간(최근 N일)", 7, 60, 14, key='shap_horizon')

                    item_daily = sdf[sdf['item'] == shap_item].groupby('date').agg(
                        sales_qty=('sales_qty', 'sum'),
                        is_weekend=('is_weekend', 'first'),
                        is_holiday=('is_holiday', 'first'),
                        is_vacation=('is_vacation', 'first'),
                        season_period=('season_period', 'first'),
                        precip_type=('precip_type', 'first'),
                        temperature=('temperature', 'first'),
                    ).sort_index().asfreq('D')
                    item_daily['sales_qty'] = item_daily['sales_qty'].fillna(0.0)
                    item_daily = item_daily.ffill()

                    if len(item_daily) < shap_horizon + 60:
                        st.warning("모델 학습에 필요한 데이터(최소 60일 + 분석기간)가 부족합니다.")
                    else:
                        X = build_shap_features(item_daily)
                        y = item_daily['sales_qty']

                        X_train, X_test = X.iloc[:-shap_horizon], X.iloc[-shap_horizon:]
                        y_train, y_test = y.iloc[:-shap_horizon], y.iloc[-shap_horizon:]

                        with st.spinner("모델 학습 및 SHAP 값 계산 중..."):
                            model = train_shap_model(X_train, y_train)
                            shap_values, base_value = explain_with_shap(model, X_test)
                            pred_test = model.predict(X_test)

                        st.markdown("#### 전역 특징 중요도 (평균 |SHAP|)")
                        importance = pd.Series(
                            np.abs(shap_values).mean(axis=0), index=X.columns
                        ).sort_values(ascending=True)
                        importance.index = [SHAP_LABELS.get(c, c) for c in importance.index]
                        fig6a = px.bar(
                            importance, orientation='h',
                            labels={'value': '평균 |SHAP| (판매량 영향력)', 'index': '피처'},
                            title=f"'{shap_item}' 예측에 대한 피처별 평균 기여도"
                        )
                        st.plotly_chart(fig6a, use_container_width=True)

                        st.markdown("#### 날짜별 예측 분해")
                        sel_date = st.selectbox(
                            "날짜 선택", list(X_test.index), key='shap_date',
                            format_func=lambda d: d.strftime('%Y-%m-%d (%a)')
                        )
                        idx = list(X_test.index).index(sel_date)
                        contribs = pd.Series(shap_values[idx], index=X.columns)
                        pred = pred_test[idx]
                        actual = y_test.iloc[idx]

                        briefing = format_shap_briefing(
                            shap_item, sel_date.strftime('%Y-%m-%d'), base_value, contribs, pred, actual
                        )
                        st.chat_message("assistant").write(briefing)

                        contrib_df = pd.DataFrame({
                            '피처': [SHAP_LABELS.get(c, c) for c in contribs.index],
                            'SHAP 기여도': contribs.values,
                        })
                        contrib_df = contrib_df.reindex(
                            contrib_df['SHAP 기여도'].abs().sort_values(ascending=False).index
                        )
                        fig6b = px.bar(
                            contrib_df, x='SHAP 기여도', y='피처', orientation='h',
                            color='SHAP 기여도', color_continuous_scale='RdBu', color_continuous_midpoint=0,
                            title=f"{sel_date.strftime('%Y-%m-%d')} 예측 기여도 분해 (기저 {base_value:.1f}개)"
                        )
                        st.plotly_chart(fig6b, use_container_width=True)

                        with st.expander("📋 기여도 상세 표"):
                            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                            st.caption(
                                f"검증: 기저치({base_value:.1f}) + 전체 기여도 합({contribs.sum():.1f}) "
                                f"= 모델 예측치({pred:.1f})"
                            )
            else:
                st.info("👈 이벤트 통합 데이터셋 CSV를 업로드하면 SHAP 기여도 분석을 실행할 수 있습니다.")

        # 탭 5: What-if 채팅
        with tab_whatif:
            st.subheader("💬 What-if 채팅 — 이벤트 조건별 예상 판매량")
            st.caption(
                "이벤트 조건(비·눈·공휴일·방학·시즌)을 체크하면 예상 판매량과 권장 생산량이 "
                "바로 바뀝니다. 문장으로 묻고 싶으면 아래 채팅을 쓰세요. "
                "※ 채팅 문구는 키워드 매칭으로 조건을 인식합니다(LLM 미연동, 계산값은 실제 모델 결과)."
            )
            whatif_file = st.file_uploader(
                "이벤트 통합 데이터셋 CSV 업로드 (date, item, sales_qty, 이벤트 컬럼 포함)",
                type=['csv'], key='whatif_csv'
            )

            if whatif_file is None and SAMPLE_DATA_PATH.exists():

                if st.checkbox('동봉된 예시 데이터로 둘러보기', key='whatif_sample'):

                    whatif_file = str(SAMPLE_DATA_PATH)


            if whatif_file is not None:
                wdf = load_event_dataset(whatif_file)
                required_cols = {'date', 'item', 'sales_qty', 'is_weekend', 'is_holiday',
                                  'is_vacation', 'season_period', 'precip_type'}
                missing_cols = required_cols - set(wdf.columns)

                if missing_cols:
                    st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")
                elif not ML_AVAILABLE:
                    st.warning("scikit-learn이 설치되어 있지 않아 이벤트 탄력도를 추정할 수 없습니다.")
                else:
                    wi_item = st.selectbox("대상 품목", sorted(wdf['item'].unique()), key='whatif_item')

                    item_daily = wdf[wdf['item'] == wi_item].groupby('date').agg(
                        sales_qty=('sales_qty', 'sum'),
                        is_weekend=('is_weekend', 'first'),
                        is_holiday=('is_holiday', 'first'),
                        is_vacation=('is_vacation', 'first'),
                        season_period=('season_period', 'first'),
                        precip_type=('precip_type', 'first'),
                    ).sort_index().asfreq('D')
                    item_daily['sales_qty'] = item_daily['sales_qty'].fillna(0.0)
                    item_daily = item_daily.ffill()

                    if len(item_daily) < 30:
                        st.warning("모델 학습에 필요한 데이터(최소 30일)가 부족합니다.")
                    else:
                        events_full = build_event_dummies(item_daily)
                        beta = fit_event_elasticity(item_daily['sales_qty'], events_full)
                        base_forecast = float(np.clip(
                            predict_holt_trend(item_daily['sales_qty'].values, 1)[0], 0, None
                        ))

                        st.markdown("##### 🔧 이벤트 조건 선택")
                        st.caption("체크하면 아래 결과가 **바로** 바뀝니다. 채팅은 문장으로 묻고 싶을 때 쓰세요.")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            chk_rain = st.checkbox("🌧 비", key='wi_rain')
                            chk_snow = st.checkbox("❄ 눈", key='wi_snow')
                        with c2:
                            chk_holiday = st.checkbox("🎌 공휴일", key='wi_holiday')
                            chk_weekend = st.checkbox("📅 주말", key='wi_weekend')
                        with c3:
                            chk_vacation = st.checkbox("🏫 방학", key='wi_vacation')
                        with c4:
                            season_choice = st.selectbox(
                                "시즌", ['없음', '크리스마스', '수능', '밸런타인', '추석'], key='wi_season'
                            )
                        tau_wi = st.slider(
                            "게이팅 임계치 τ", 0.0, 0.3, 0.10, 0.01, key='wi_tau',
                            help="이벤트 영향이 이 값보다 작으면 무시하고 기저 예측을 씁니다. "
                                 "약한 이벤트까지 매번 반영하면 오히려 예측이 흔들리기 때문입니다. "
                                 "0으로 두면 모든 이벤트를 반영합니다."
                        )

                        def whatif_events(flags=None):
                            """체크박스 상태(+선택적으로 채팅에서 읽은 조건)를 합쳐 이벤트 벡터를 만든다."""
                            f = flags or {k: False for k in WHATIF_KEYWORD_MAP}
                            return {
                                'is_weekend': float(chk_weekend or f['is_weekend']),
                                'is_holiday': float(chk_holiday or f['is_holiday']),
                                'is_vacation': float(chk_vacation or f['is_vacation']),
                                'is_rain': float(chk_rain or f['is_rain']),
                                'is_snow': float(chk_snow or f['is_snow']),
                                'is_christmas': float(season_choice == '크리스마스' or f['is_christmas']),
                                'is_suneung': float(season_choice == '수능' or f['is_suneung']),
                                'is_valentine': float(season_choice == '밸런타인' or f['is_valentine']),
                                'is_chuseok': float(season_choice == '추석' or f['is_chuseok']),
                            }

                        # ---------- 즉시 결과 ----------
                        # 체크박스를 건드리는 순간 바로 보이도록, 채팅과 무관하게 항상 계산해 띄운다.
                        live_events = whatif_events()
                        live = whatif_compute(beta, base_forecast, live_events, tau_wi)
                        live_active = [SHAP_LABELS.get(k, k) for k, v in live_events.items() if v]

                        st.markdown("---")
                        r1, r2, r3 = st.columns(3)
                        r1.metric("기저 예측 (이벤트 없음)", f"{base_forecast:.0f}개")
                        r2.metric("조건 반영 예상 판매량", f"{live['final']:.0f}개",
                                  delta=f"{live['pct']:+.1f}%")
                        r3.metric("권장 생산량 (안전재고 10%)", f"{live['production']}개")

                        if not live_active:
                            st.caption("위에서 조건을 체크하면 예상 판매량이 어떻게 달라지는지 바로 보여드립니다.")
                        else:
                            st.caption(f"선택한 조건: {', '.join(live_active)}")

                            # 어떤 이벤트가 얼마나 밀어올리고 끌어내렸는지 보여준다.
                            # 이게 없으면 "비를 켰는데 숫자가 그대로"인 이유를 알 수 없다.
                            contrib_rows = [
                                {'조건': SHAP_LABELS.get(k, k),
                                 '판매량 영향': f"{v * 100:+.1f}%"}
                                for k, v in sorted(live['contribs'].items(),
                                                    key=lambda kv: -abs(kv[1]))
                            ]
                            cc1, cc2 = st.columns([1, 1.3])
                            with cc1:
                                st.dataframe(pd.DataFrame(contrib_rows),
                                              use_container_width=True, hide_index=True)
                            with cc2:
                                if live['gate_on']:
                                    st.success(
                                        f"이벤트 강도 합계 **{live['score']:+.2f}** 가 임계치 "
                                        f"τ={tau_wi:.2f} 이상이라 예측에 반영했습니다."
                                    )
                                else:
                                    st.warning(
                                        f"이벤트 강도 합계 **{live['score']:+.2f}** 가 임계치 "
                                        f"τ={tau_wi:.2f} 보다 작아 **반영하지 않았습니다.** "
                                        f"그래서 위 숫자가 기저 예측과 같습니다.\n\n"
                                        f"영향이 작다고 판단한 것이므로 정상 동작입니다. "
                                        f"그래도 반영해 보고 싶으면 **τ를 "
                                        f"{max(abs(live['score']) - 0.01, 0.0):.2f} 이하로** 내려보세요."
                                    )

                        # ---------- 채팅 ----------
                        st.markdown("---")
                        st.markdown("##### 💬 문장으로 물어보기")

                        if 'whatif_messages' not in st.session_state:
                            st.session_state.whatif_messages = []

                        if st.button("🗑 대화 초기화", key='whatif_clear'):
                            st.session_state.whatif_messages = []
                            st.rerun()

                        for msg in st.session_state.whatif_messages:
                            st.chat_message(msg['role']).write(msg['content'])

                        user_text = st.chat_input(
                            f"'{wi_item}'에 대해 물어보세요 (예: 내일 비 오고 방학이면?)"
                        )
                        if user_text:
                            st.session_state.whatif_messages.append({'role': 'user', 'content': user_text})
                            text_flags = parse_events_llm(user_text)
                            if text_flags is None:
                                text_flags = parse_whatif_keywords(user_text)

                            events = whatif_events(text_flags)
                            r = whatif_compute(beta, base_forecast, events, tau_wi)
                            active = [SHAP_LABELS.get(k, k) for k, v in events.items() if v]
                            cond_str = ', '.join(active) if active else '특별한 이벤트 없음'
                            reply = whatif_reply_text(wi_item, cond_str, base_forecast, r, tau_wi)

                            ctx = {
                                'item': wi_item,
                                'cond_str': cond_str,
                                'base': base_forecast,
                                'final': r['final'],
                                'pct': r['pct'],
                                'production': r['production'],
                                'score': r['score'],
                                'tau': tau_wi,
                                'gate': '발동' if r['gate_on'] else '미발동',
                            }

                            with st.chat_message('user'):
                                st.markdown(user_text)

                            with st.chat_message('assistant'):
                                try:
                                    streamed = st.write_stream(stream_reply(ctx))
                                except Exception as e:
                                    print(f"[whatif_llm] 스트리밍 실패: {e}")
                                    streamed = None

                                if not streamed:
                                    st.markdown(reply)
                                    streamed = reply

                            st.session_state.whatif_messages.append(
                                {'role': 'assistant', 'content': streamed}
                            )
            else:
                st.info("👈 이벤트 통합 데이터셋 CSV를 업로드하면 What-if 채팅을 사용할 수 있습니다.")

        # 탭 6: 이벤트 우선 게이팅(H1) 검증 — 개발자 전용, 맨 뒤에 배치
        with tab_h1:
            st.error(
                "⚙️ **개발자 전용 탭입니다.** 모델이 제대로 동작하는지 검증하는 화면이라 "
                "매장 운영에는 필요하지 않습니다. 그냥 지나치셔도 됩니다."
            )
            st.subheader("🌦 이벤트 우선 게이팅 (Layer 2, H1) — Walk-forward 백테스트")
            st.caption(
                "공휴일·방학·시즌·날씨 이벤트를 반영한 예측이, 이벤트를 무시한 기저 시계열 예측보다 "
                "실제로 더 정확한지 최근 기간을 떼어내 검증합니다 (가설 H1). "
                "미래 이벤트는 알 수 없으므로 '미래 예측'이 아니라 과거 데이터로 백테스트합니다."
            )
            event_file = st.file_uploader(
                "이벤트 통합 데이터셋 CSV 업로드 (date, item, sales_qty, is_weekend, is_holiday, "
                "is_vacation, season_period, precip_type 컬럼 포함)",
                type=['csv'], key='event_csv'
            )

            if event_file is None and SAMPLE_DATA_PATH.exists():

                if st.checkbox('동봉된 예시 데이터로 둘러보기', key='event_sample'):

                    event_file = str(SAMPLE_DATA_PATH)


            if event_file is not None:
                edf = load_event_dataset(event_file)
                required_cols = {'date', 'item', 'sales_qty', 'is_weekend', 'is_holiday',
                                  'is_vacation', 'season_period', 'precip_type'}
                missing_cols = required_cols - set(edf.columns)

                if missing_cols:
                    st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")
                elif not ML_AVAILABLE:
                    st.warning("scikit-learn이 설치되어 있지 않아 이벤트 탄력도를 추정할 수 없습니다.")
                else:
                    ev_items = sorted(edf['item'].unique())
                    ev_item = st.selectbox("품목 선택", ev_items, key='event_item')
                    horizon = st.slider("백테스트 검증 기간(일)", 7, 28, 14, key='event_horizon')
                    tau = st.slider(
                        "이벤트 게이팅 임계치 τ (이 값을 넘는 이벤트 강도만 예측에 반영)",
                        0.0, 0.3, 0.10, 0.01, key='event_tau'
                    )

                    item_daily = edf[edf['item'] == ev_item].groupby('date').agg(
                        sales_qty=('sales_qty', 'sum'),
                        is_weekend=('is_weekend', 'first'),
                        is_holiday=('is_holiday', 'first'),
                        is_vacation=('is_vacation', 'first'),
                        season_period=('season_period', 'first'),
                        precip_type=('precip_type', 'first'),
                    ).sort_index()

                    if len(item_daily) < horizon + 30:
                        st.warning("검증에 필요한 데이터(최소 학습 30일 + 검증기간)가 부족합니다.")
                    else:
                        train = item_daily.iloc[:-horizon]
                        test = item_daily.iloc[-horizon:]

                        events = build_event_dummies(item_daily)
                        beta = fit_event_elasticity(train['sales_qty'], events.loc[train.index])

                        base_fc = predict_prophet(train['sales_qty'], horizon, freq='D')
                        base_fc = np.clip(base_fc, 0, None)

                        # Layer 2 본선: 이벤트를 Prophet 회귀변수로 기저모델에 직접 투입
                        event_fc = predict_prophet_with_events(
                            train['sales_qty'], events.loc[train.index], events.loc[test.index], freq='D'
                        )
                        event_fc = np.clip(event_fc, 0, None)

                        # 대조군(ablation): 기존 곱셈형 사후 게이팅
                        gated_fc, event_score, gate_on = apply_event_gating(
                            base_fc, beta, events.loc[test.index], tau
                        )

                        actual = test['sales_qty'].values
                        wape_base = compute_wape(actual, base_fc)
                        wape_event = compute_wape(actual, event_fc)
                        wape_gated = compute_wape(actual, gated_fc)

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("기저 예측(Layer 1) WAPE", f"{wape_base:.1f}%")
                        m2.metric("이벤트 회귀(H1) WAPE", f"{wape_event:.1f}%",
                                  delta=f"{wape_event - wape_base:+.1f}%p", delta_color="inverse")
                        m3.metric("곱셈형 게이팅(대조군) WAPE", f"{wape_gated:.1f}%",
                                  delta=f"{wape_gated - wape_base:+.1f}%p", delta_color="inverse")
                        m4.metric("게이팅 발동 일수", f"{int(gate_on.sum())} / {horizon}일")

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
                        fig3.add_trace(go.Scatter(x=test.index, y=actual, name='실제',
                                                   line=dict(color='black', width=3)))
                        fig3.add_trace(go.Scatter(x=test.index, y=base_fc, name='기저 예측(Layer 1)',
                                                   line=dict(color='blue', dash='dot')))
                        fig3.add_trace(go.Scatter(x=test.index, y=event_fc, name='이벤트 회귀 예측(H1)',
                                                   line=dict(color='red', width=3)))
                        fig3.add_trace(go.Scatter(x=test.index, y=gated_fc, name='곱셈형 게이팅(대조군)',
                                                   line=dict(color='gray', dash='dash', width=1.5)))
                        for d, on in zip(test.index, gate_on):
                            if on:
                                fig3.add_vrect(x0=d - pd.Timedelta(hours=12), x1=d + pd.Timedelta(hours=12),
                                               fillcolor='orange', opacity=0.15, line_width=0)
                        fig3.update_layout(
                            height=450, hovermode='x unified',
                            title=f"{ev_item} — 이벤트 게이팅 백테스트 (주황 음영 = 게이팅 발동일)"
                        )
                        st.plotly_chart(fig3, use_container_width=True)

                        with st.expander("📊 추정된 이벤트 탄력도(β) 보기"):
                            beta_df = pd.DataFrame({'이벤트': list(beta.keys()), '탄력도(β)': list(beta.values())})
                            beta_df = beta_df.reindex(
                                beta_df['탄력도(β)'].abs().sort_values(ascending=False).index
                            )
                            st.dataframe(beta_df, use_container_width=True, hide_index=True)
                            st.caption("β > 0: 해당 이벤트일 때 판매량 증가 경향 / β < 0: 감소 경향 "
                                       "(학습 구간 기준 선형회귀 추정치, Σβₖ·eₖ가 이벤트 강도 점수)")
            else:
                st.info("👈 방학·공휴일·날씨·시즌 이벤트가 포함된 통합 데이터셋 CSV를 업로드하면 H1 가설을 검증할 수 있습니다.")



else:
    st.info("👈 왼쪽 사이드바에서 빵집 판매내역 CSV 파일을 업로드해주세요.")
