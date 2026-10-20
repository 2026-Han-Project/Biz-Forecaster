import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from pathlib import Path
from agent.whatif_llm import parse_events_llm, stream_reply

# --- 수요예측 엔진: 매입 -> 역레시피 -> 진짜 수요 복원 -> 수준, 요일 학습 -> 시즌, 이벤트 -> 권장 생산량 ---
# 구현체는 forecast/ 패키지(Streamlit 비의존). app.py 는 화면만 담당한다.
from forecast import (
    Config, WEEKDAY_NAMES, InputError, FILE_NAMES, load_inputs, fit, next_open_days,
    cumulative_share,
)

# --- 팀원 C 파트: 이상탐지(SPC 관리도), 성능평가 모듈 ---
from detect.control_charts import (
    shewhart_3sigma, ewma_chart, cusum_chart, weekday_residual,
)
# 탐지 결과를 '재고를 어떻게 하라'는 말로 옮기는 계층.
# 통계 용어(3시그마, EWMA, CUSUM)는 화면에 나가지 않는다.
from detect.alerts import build_alerts, build_context, trend_summary, weekday_guide

# --- 계정 인증 (소상공인별 로그인) ---
from auth.ui import (
    require_login, render_sidebar_account, render_account_page, ACCOUNT_VIEW_KEY,
)
from auth import store as auth_store

# 시각화 라이브러리
import plotly.graph_objects as go
import plotly.express as px

# --- 머신러닝 라이브러리 (POS 판매데이터 탭: SHAP, H1, 설치 여부 확인) ---
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# --- Prophet 라이브러리 (H1 백테스트 기저모델, 설치 여부 확인) ---
try:
    import logging as _logging
    from prophet import Prophet

    _logging.getLogger('cmdstanpy').setLevel(_logging.WARNING)
    _logging.getLogger('prophet').setLevel(_logging.WARNING)

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# --- SHAP 라이브러리 (설치 여부 확인) ---
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Biz Forecaster - 소상공인 AI 수요예측", layout="wide")

# -----------------------------------------------------------------------------
# 1-1. 로그인 게이트
# -----------------------------------------------------------------------------
# 소상공인마다 자기 매장 데이터만 보도록, 분석 화면 전체를 로그인 뒤로 숨긴다.
# 미로그인 상태에서는 require_login()이 인증 화면을 그리고 None을 돌려주며,
# st.stop()으로 아래 코드를 아예 실행하지 않는다 (데이터가 그려질 여지 자체를 없앤다).
# 동봉된 예시 데이터. 데이터가 없어도 화면을 둘러볼 수 있게 한다.
SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "integrated_dataset.csv"   # POS 판매데이터 탭용
SAMPLE_INPUT_DIR = Path(__file__).parent / "data" / "sample"                   # 생산량 추천용 입력 5종

CURRENT_USER = require_login()
if CURRENT_USER is None:
    st.stop()

render_sidebar_account()

# 계정 설정 화면이 열려 있으면 분석 화면 대신 그것만 그린다.
if st.session_state.get(ACCOUNT_VIEW_KEY):
    render_account_page()
    st.stop()


def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()


# -----------------------------------------------------------------------------
# 2. 시계열 기저모델 (H1 백테스트 탭 전용)
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

def predict_prophet(dated_series, periods, freq='D'):
    """Layer 1 기저모델(Prophet). dated_series는 DatetimeIndex를 가진 pd.Series여야
    실제 달력(연간, 주간 계절성)을 학습할 수 있다. 실패, 미설치, 데이터 부족 시
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
    """Layer 2(H1) - 이벤트를 사후 곱셈 보정이 아니라 Prophet 회귀변수로
    기저모델 안에 직접 투입한다.

    기존 곱셈형 게이팅 `y_hat_basex(1+sumbeta_ke_k)`은 Prophet이 주간 계절성으로 이미
    학습한 주말 효과를 다시 곱해 이벤트 효과를 **이중 계상**한다(실측상
    beta_weekend가 품목마다 +0.35~+0.43로 추정되어, 주말마다 예측치를 35~43%
    추가 증폭시키고 있었다). `add_regressor()`로 넣으면 추세, 계절성, 이벤트가
    한 번의 적합에서 동시에 분리 추정되므로 이 중복이 구조적으로 사라진다.

    실측(일별, 검증 14일, 4품목 평균 WAPE): 기저 31.7% / 곱셈형 게이팅 35.8%
    / 이벤트 회귀 22.5%.

    미설치, 데이터 부족, 학습 실패 시 predict_prophet(이벤트 미반영)으로 폴백한다.
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

# -----------------------------------------------------------------------------
# 5. 이벤트 우선 게이팅 (Layer 2, H1) - 통합 이벤트 데이터셋 기반 Walk-forward 백테스트
# -----------------------------------------------------------------------------
# 데이터셋의 미래(예: 다음 5주)의 실제 공휴일, 날씨는 알 수 없으므로, '미래 예측'이 아니라
# 최근 N일을 떼어내 실제값과 비교하는 백테스트로 H1(이벤트 우선 예측이 기저 예측보다 정확한가)을 검증한다.

@st.cache_data
def load_event_dataset(uploaded_file):
    return pd.read_csv(uploaded_file, parse_dates=['date'])


def build_event_dummies(daily):
    """일별 이벤트 플래그(요일, 공휴일, 방학, 시즌, 날씨)를 0/1 더미로 변환."""
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
    """이벤트별 수요 탄력도(beta_k)를 추정한다.

    국소 기저치(14일 중심 이동중앙값) 대비 실제값의 상대 편차를 이벤트 더미에 회귀시켜,
    'y_hat_final = y_hat_base x (1 + sum beta_k, e_k)' 융합식의 beta_k를 구한다.
    """
    baseline = qty.rolling(14, center=True, min_periods=7).median()
    baseline = baseline.bfill().ffill()
    pct_dev = (qty - baseline) / baseline.replace(0, np.nan)
    pct_dev = pct_dev.fillna(0.0)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(events.values, pct_dev.values)
    return dict(zip(events.columns, reg.coef_))


def apply_event_gating(base_forecast, beta, test_events, tau):
    """이벤트 강도 점수가 임계치 tau를 넘는 날에만 이벤트 조정치를 예측에 반영(게이팅)."""
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
# 7. 실시간 이상탐지 - SPC 관리도 3종 + Isolation Forest
# -----------------------------------------------------------------------------
# 구현체는 detect/ 패키지로 분리했다(detect/control_charts.py, detect/evaluation.py).
# app.py는 화면만 담당하고, 탐지 로직은 UI 없이 단독으로 테스트할 수 있게 떼어낸 것이다.
#
# 팀원 C 파트에서 이번에 보강한 것
#   - CUSUM 관리도 추가 (기존에는 3시그마, EWMA뿐이라 점진적 드리프트를 놓치고 있었다)
#   - 요일 계절성을 제거한 잔차 관리도 옵션 (SPC의 정상상태 가정을 맞춰준다)
#   - 오탐/미탐 트레이드오프 정량화 (detect/evaluation.py, 탭 8)
#
# 아래 세 함수는 기존 호출부와의 호환을 위해 남겨둔 얇은 래퍼다.

def rolling_3sigma_flags(series, window=28, min_periods=10, L=3.0, non_negative=True):
    """3시그마(Shewhart) 관리도 - detect.control_charts.shewhart_3sigma 래퍼."""
    return shewhart_3sigma(series, L=L, window=window,
                            min_periods=min_periods, non_negative=non_negative)


def ewma_flags(series, lam=0.2, L=3, window=28, min_periods=10, non_negative=True):
    """EWMA 관리도 - detect.control_charts.ewma_chart 래퍼."""
    return ewma_chart(series, lam=lam, L=L, window=window,
                       min_periods=min_periods, non_negative=non_negative)


def cusum_flags(series, k=0.5, h=5.0, window=28, min_periods=10):
    """CUSUM 관리도 - detect.control_charts.cusum_chart 래퍼 (이번에 새로 추가)."""
    return cusum_chart(series, k=k, h=h, window=window, min_periods=min_periods)


# -----------------------------------------------------------------------------
# 8. SHAP 기여도 분석 (XAI)
# -----------------------------------------------------------------------------
# build_event_dummies()로 만든 이벤트 피처에 기온, 추세를 더해 RandomForest로 학습하고,
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
# 9. 생산량 추천 - 입력 파일, 환경 변수, 화면
# -----------------------------------------------------------------------------
# 예측 계산은 forecast/ 패키지가 한다. 여기서는 입력을 모으고 결과를 사장님 말로 옮긴다.

INPUT_HELP = {
    'products': "제품명, 판매가, 원가, 회수가, 평소 생산량 - 1회 입력. "
                "회수가 = 다음 날 할인가 등, 없으면 0",
    'recipe': "제품, 재료, 1개당 재료량 - 1회 입력. 핵심 재료 2~5개면 충분",
    'purchases': "날짜, 재료, 수량 - 과거분 일괄. 장부, 거래명세서에서 그대로 옮기면 됨",
    'daily_log': "날짜, 제품, 결과, (생산량) - 매일 마감 시. "
                 "결과: 3 = 3개 남음, 15:30 = 15시 30분 품절, 0 = 마감 무렵 다 팔림, "
                 "빈칸 = 입력 안 함. 생산량은 평소와 다를 때만 입력",
    'events': "날짜, 태그(holiday, rain, event) - 선택",
}
TAG_LABELS = {'holiday': '공휴일', 'rain': '비', 'event': '행사'}
WHATIF_KEYWORDS = {
    'holiday': ['공휴일', '연휴', '휴일', '명절', '빨간날', '추석', '설날'],
    'rain': ['비 오', '비가', '비오', '우천', '장마', '강수'],
    'event': ['행사', '축제', '이벤트', '크리스마스', '밸런타인', '발렌타인', '수능'],
}
CONFIG_FILE = 'config.json'


def tag_label(tag):
    return TAG_LABELS.get(tag, tag)


@st.cache_resource(show_spinner=False, max_entries=8)
def build_model(blobs, cfg_json):
    """입력 파일(바이트)과 설정이 같으면 다시 학습하지 않는다."""
    inputs = load_inputs({k: io.BytesIO(b) for k, b in blobs})
    return fit(inputs, Config.from_dict(json.loads(cfg_json)))


def render_config_editor(base: Config, user_dir: Path) -> Config:
    """환경 변수 표를 그대로 화면에 옮긴 편집기. 바꾼 값은 즉시 예측에 반영된다."""
    st.subheader("환경 변수 (Config)")
    st.caption("바꾸면 다른 탭의 예측이 바로 다시 계산됩니다. 매장에 맞는 값을 찾았으면 **저장**하세요.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**영업**")
        open_hour = st.number_input("open_hour - 영업 시작(시)", 0, 23, base.open_hour, key='cfg_open')
        close_hour = st.number_input("close_hour - 영업 마감(시)", 1, 24, base.close_hour, key='cfg_close')
        closed = st.multiselect("closed_weekdays - 정기 휴무 요일", WEEKDAY_NAMES,
                                [WEEKDAY_NAMES[d] for d in base.closed_weekdays], key='cfg_closed')
        st.markdown("**학습 속도**")
        alpha = st.slider("alpha - 최근 변화 반응 속도", 0.05, 0.8, float(base.alpha), 0.05, key='cfg_alpha',
                          help="높이면 최근 변화에 빨리 반응합니다 (대신 노이즈도 커짐)")
        k_cred = st.number_input("k_cred - 관측을 믿기까지 필요한 입력 일수", 0.0, 60.0,
                                 float(base.k_cred), 1.0, key='cfg_kcred',
                                 help="낮을수록 사전값에서 관측으로 빨리 전환합니다. k=5면 5일에 50:50, 20일에 관측 80%")
        k_weekday = st.number_input("k_weekday - 요일 계수 보수성", 0.0, 60.0,
                                    float(base.k_weekday), 1.0, key='cfg_kwd',
                                    help="요일 편차가 과해 보이면 높이세요")
    with c2:
        st.markdown("**품절 보정**")
        max_uplift = st.number_input("max_uplift - 품절 보정 상한(배)", 1.0, 5.0,
                                     float(base.max_uplift), 0.1, key='cfg_maxup')
        close_uplift = st.number_input("close_uplift - 마감 직전 품절 시 보정(배)", 1.0, 2.0,
                                       float(base.close_uplift), 0.01, key='cfg_closeup')
        st.markdown("**역레시피, 사전값**")
        loss_rate = st.number_input("loss_rate - 재료 로스(반죽 손실, 실패, 시식)", 0.0, 0.5,
                                    float(base.loss_rate), 0.01, key='cfg_loss')
        prior_waste = st.number_input("prior_waste - 초기 단계 안 팔린 비율 가정", 0.0, 0.6,
                                      float(base.prior_waste), 0.01, key='cfg_pwaste')
        bom_window_days = st.number_input("bom_window_days - 역레시피 계산 기간(일)", 7, 120,
                                          int(base.bom_window_days), 7, key='cfg_bomwin')
        bom_lambda = st.number_input("bom_lambda - 평소 생산 구성비를 믿는 정도", 0.0, 10.0,
                                     float(base.bom_lambda), 0.1, key='cfg_bomlam')
    with c3:
        st.markdown("**권장 생산량**")
        default_cv = st.number_input("default_cv - 데이터 부족 시 수요 변동계수", 0.05, 1.0,
                                     float(base.default_cv), 0.05, key='cfg_cv')
        sl_bounds = st.slider("sl_bounds - 목표 서비스수준 범위", 0.30, 0.99,
                              tuple(base.sl_bounds), 0.01, key='cfg_sl')
        st.markdown("**시즌, 이벤트**")
        season_bounds = st.slider("season_bounds - 시즌 보정 하한, 상한", 0.5, 2.0,
                                  tuple(base.season_bounds), 0.05, key='cfg_season')
        ev_df = st.data_editor(
            pd.DataFrame({'태그': list(base.event_factors), '배수': list(base.event_factors.values())}),
            num_rows='dynamic', hide_index=True, width='stretch', key='cfg_events',
            column_config={'배수': st.column_config.NumberColumn(min_value=0.1, max_value=5.0, step=0.05)},
        )
        st.caption("event_factors - events.csv 의 태그별 배수 (holiday 공휴일, rain 비, event 행사)")

    p1, p2 = st.columns([1.6, 1])
    with p1:
        st.markdown("**hourly_profile - 시간대별 판매 비중** (품절 보정 강도를 결정하므로 매장에 맞게 조정)")
        hours = list(range(int(open_hour), int(close_hour)))
        prof_df = pd.DataFrame([[base.hourly_profile.get(h, 0.0) for h in hours]],
                               columns=[f"{h}시" for h in hours])
        prof_df = st.data_editor(prof_df, hide_index=True, width='stretch',
                                 key=f'cfg_profile_{open_hour}_{close_hour}')
    with p2:
        st.markdown("**weekday_prior - 사장님의 요일 감각** (예: 토요일 1.2)")
        wd_df = st.data_editor(pd.DataFrame([base.weekday_prior], columns=WEEKDAY_NAMES),
                               hide_index=True, width='stretch', key='cfg_wdprior')

    cfg = Config(
        open_hour=int(open_hour), close_hour=int(close_hour),
        closed_weekdays=[WEEKDAY_NAMES.index(d) for d in closed],
        hourly_profile={h: float(v) if pd.notna(v) else 0.0
                        for h, v in zip(hours, prof_df.iloc[0])},
        weekday_prior=[float(v) if pd.notna(v) else 0.0 for v in wd_df.iloc[0]],
        alpha=float(alpha), k_cred=float(k_cred), k_weekday=float(k_weekday),
        max_uplift=float(max_uplift), close_uplift=float(close_uplift),
        loss_rate=float(loss_rate), prior_waste=float(prior_waste),
        bom_window_days=int(bom_window_days), bom_lambda=float(bom_lambda),
        default_cv=float(default_cv), sl_bounds=sl_bounds,
        event_factors={str(t).strip().lower(): float(f) for t, f in zip(ev_df['태그'], ev_df['배수'])
                       if str(t).strip() and t is not None and pd.notna(f)},
        season_bounds=season_bounds,
    )
    errs = cfg.validate()
    for e in errs:
        st.error(e)

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("이 설정 저장", disabled=bool(errs)):
        cfg.save(user_dir / CONFIG_FILE)
        st.session_state.cfg_base = cfg.to_dict()
        st.success("저장했습니다. 다음 로그인 때도 이 설정으로 계산합니다.")
    if b2.button("기본값으로"):
        for k in [k for k in st.session_state if str(k).startswith('cfg_') and k != 'cfg_base']:
            del st.session_state[k]
        st.session_state.cfg_base = Config().to_dict()
        st.rerun()

    return base if errs else cfg


def explain_prediction(model, r):
    """예측 1건을 사장님이 읽는 두세 문장으로."""
    prod = model.inputs.products.loc[r['product']]
    lv = model.levels.loc[r['product']]
    when = f"{r['date'].month}월 {r['date'].day}일({r['weekday']})"
    lines = [f"{when} '{r['product']}'은 약 **{r['mu']:.0f}개** 팔릴 것으로 보입니다."]

    why = []
    if abs(r['weekday_coef'] - 1) >= 0.05:
        why.append(f"{r['weekday']}요일은 평소의 {r['weekday_coef']:.0%} 수준")
    if abs(r['season'] - 1) >= 0.03:
        why.append(f"작년 이맘때 생산 흐름 기준 시즌 보정 {r['season']:.2f}배")
    if r['tags']:
        why.append(f"{', '.join(tag_label(t) for t in r['tags'])} 배수 {r['event']:.2f}")
    if why:
        lines.append("반영한 것: " + ", ".join(why) + ".")

    sl = r['service_level']
    if sl >= 0.9:
        style = (f"남아도 손해가 적은 품목이라(회수가 {prod['salvage']:,.0f}원) 넉넉하게 - "
                 f"품절이 나지 않을 확률 {sl:.0%}")
    elif sl <= 0.7:
        style = f"버리면 손해가 큰 품목이라 빠듯하게 - 품절이 나지 않을 확률 {sl:.0%}"
    else:
        style = f"품절이 나지 않을 확률 {sl:.0%}"
    lines.append(f"{style}를 목표로 **{r['production']}개**를 권합니다.")

    if lv['n_obs'] < model.cfg.k_cred * 4:
        lines.append(f"아직 마감 기록이 {int(lv['n_obs'])}일치라, 매입장부로 추정한 사전값을 "
                     f"{1 - lv['weight']:.0%} 섞었습니다.")
    return " ".join(lines)


def prediction_chain(model, r):
    """사전값, 학습 수준, 요일, 시즌, 이벤트, 권장 생산량 순서의 단계별 숫자."""
    lv = model.levels.loc[r['product']]
    cfg = model.cfg
    rows = [
        ("2. 역레시피 생산 추정", f"{lv['prior_production']:.1f}개/일",
         f"{model.bom.method} ({cfg.bom_window_days}일)"),
        ("2. 사전값", f"{lv['prior']:.1f}개", f"생산량 곱하기 (1 - prior_waste {cfg.prior_waste:.0%})"),
        ("4. 관측 수준 (EWMA)", "없음" if pd.isna(lv['ewma']) else f"{lv['ewma']:.1f}개",
         f"마감 기록 {int(lv['n_obs'])}일 (품절 복원 {int(lv['n_censored'])}일), alpha {cfg.alpha}"),
        ("4. 학습 수준", f"{lv['level']:.1f}개",
         f"관측 {lv['weight']:.0%}, 사전값 {1 - lv['weight']:.0%} 비율로 섞음 (n/(n+k), k={cfg.k_cred:g})"),
        ("4. 요일 계수", f"{r['weekday_coef']:.2f}배", f"{r['weekday']}요일"),
        ("5. 시즌 보정", f"{r['season']:.2f}배",
         "작년 같은 시기 매입 흐름" if model.season_enabled else "매입 이력 1년 미만이라 미적용"),
        ("5. 이벤트 배수", f"{r['event']:.2f}배",
         ", ".join(tag_label(t) for t in r['tags']) or "없음"),
        ("5. 예상 판매량", f"{r['mu']:.1f}개", "학습 수준에 위 계수를 모두 곱한 값"),
        ("6. 목표 서비스수준", f"{r['service_level']:.0%}",
         f"(판매가 - 원가) / (판매가 - 회수가), {cfg.sl_bounds[0]:.0%}~{cfg.sl_bounds[1]:.0%} 범위로 제한"),
        ("6. 권장 생산량", f"{r['production']}개",
         f"예상 판매량에 여유분 추가 (z {r['z']:.2f}, 변동계수 {r['cv']:.2f})"),
    ]
    return pd.DataFrame(rows, columns=['단계', '값', '근거'])


def render_input_guide():
    st.info("왼쪽 사이드바에서 입력 파일을 올리거나 **동봉된 예시 데이터로 둘러보기**를 켜세요. "
            "`products.csv` 하나만 있어도 평소 생산량 기준으로 예측이 돌아갑니다.")
    st.markdown("##### 입력 변수")
    st.dataframe(pd.DataFrame([
        ('products.csv', '제품명, 판매가, 원가, 회수가, 평소 생산량', '1회', '회수가 = 다음 날 할인가 등. 없으면 0'),
        ('recipe.csv', '제품 1개당 주요 재료량', '1회', '핵심 재료 2~5개면 충분'),
        ('purchases.csv', '날짜, 재료, 수량', '과거분 일괄', '장부, 거래명세서에서 그대로 옮기면 됨'),
        ('daily_log.csv', '날짜, 제품, 결과, (생산량)', '매일 마감 시',
         '결과: 3 = 3개 남음, 15:30 = 15시 30분 품절, 0 = 마감 무렵 다 팔림, 빈칸 = 입력 안 함. '
         '생산량은 평소와 다를 때만 입력'),
        ('events.csv', '날짜, 태그', '선택', 'holiday, rain, event'),
    ], columns=['파일', '내용', '입력 주기', '비고']), hide_index=True, width='stretch')
    st.caption("매일 입력은 **제품당 값 하나**뿐입니다. 입력을 빠뜨린 날은 건너뛰고, 예측은 그대로 돌아갑니다.")


def render_plan_tab(model):
    st.subheader("권장 생산량")
    first = next_open_days(model, 1)[0]
    c1, c2 = st.columns([1, 1])
    start = c1.date_input("예측일", value=first.date(),
                          min_value=(model.origin + pd.Timedelta(days=1)).date(), key='plan_start')
    n_days = c2.slider("며칠치 볼까요", 1, 7, 1, key='plan_days')
    dates = next_open_days(model, n_days, start=start)
    st.caption(f"학습 기준일 **{model.origin:%Y-%m-%d}** - 이 날까지 들어온 매입장부와 마감 기록으로 계산했습니다.")
    for n in model.notes:
        st.info(n)

    plan = model.plan(dates)
    d0 = dates[0]
    p0 = plan[plan['date'] == d0].set_index('product')
    tags = model.event_tags(d0)
    title = f"{d0.month}월 {d0.day}일 ({WEEKDAY_NAMES[d0.weekday()]})"
    if tags:
        title += ", " + ", ".join(tag_label(t) for t in tags)
    st.markdown(f"#### {title}")

    total_q, total_usual = int(p0['production'].sum()), int(p0['usual_qty'].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("권장 생산량 합계", f"{total_q}개", delta=f"평소 대비 {total_q - total_usual:+d}개",
              delta_color='off')
    m2.metric("예상 판매량 합계", f"{p0['mu'].sum():.0f}개")
    m3.metric("학습에 쓴 마감 기록", f"{int(model.levels['n_obs'].sum())}건",
              help="빈칸으로 둔 날은 건너뜁니다.")

    table = pd.DataFrame({
        '권장 생산량': p0['production'],
        '평소 생산량': p0['usual_qty'].astype(int),
        '예상 판매량': p0['mu'].round(1),
        '품절 안 날 확률(목표)': (p0['service_level'] * 100).round().astype(int).astype(str) + '%',
        '요일 계수': p0['weekday_coef'].round(2),
        '시즌': p0['season'].round(2),
        '이벤트': p0['event'].round(2),
        '관측 비중': (model.levels.loc[p0.index, 'weight'] * 100).round().astype(int).astype(str) + '%',
    })
    table.index.name = '제품'
    st.dataframe(table, width='stretch')
    st.caption("남아도 손해가 적은 품목은 여유 있게, 버리면 손해가 큰 품목은 빠듯하게 만듭니다. "
               "**관측 비중**은 마감 기록을 얼마나 믿었는지(나머지는 매입장부로 추정한 사전값)입니다.")

    if n_days > 1:
        st.markdown("##### 날짜별 권장 생산량")
        wide = plan.assign(날짜=plan['date'].map(lambda d: f"{d:%m/%d}({WEEKDAY_NAMES[d.weekday()]})")) \
                   .pivot(index='product', columns='날짜', values='production')
        wide = wide[[f"{d:%m/%d}({WEEKDAY_NAMES[d.weekday()]})" for d in dates]]
        wide.index.name = '제품'
        st.dataframe(wide, width='stretch')

    needs = model.ingredient_needs(plan)
    if not needs.empty:
        st.markdown("##### 필요한 재료량 (권장 생산량 기준, 재료 로스 포함)")
        needs.index = [f"{d:%m/%d}({WEEKDAY_NAMES[d.weekday()]})" for d in needs.index]
        st.dataframe(needs.T.round(2), width='stretch')

    export = plan.assign(
        날짜=plan['date'].dt.strftime('%Y-%m-%d'),
        이벤트=plan['tags'].map(lambda t: ", ".join(tag_label(x) for x in t)),
    ).rename(columns={'product': '제품', 'weekday': '요일', 'production': '권장 생산량',
                      'mu': '예상 판매량', 'service_level': '목표 서비스수준',
                      'usual_qty': '평소 생산량', 'weekday_coef': '요일 계수',
                      'season': '시즌', 'event': '이벤트 배수'})
    export = export[['날짜', '요일', '제품', '권장 생산량', '평소 생산량', '예상 판매량',
                     '목표 서비스수준', '요일 계수', '시즌', '이벤트', '이벤트 배수']]
    st.download_button("엑셀 다운로드", data=to_excel(export.round(3)),
                       file_name=f"생산계획_{d0:%Y%m%d}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ---------- 제품별 자세히 ----------
    st.markdown("---")
    st.subheader("제품별 자세히 보기")
    prod = st.selectbox("제품", list(model.inputs.products.index), key='plan_product')
    r = model.predict(prod, d0)
    st.chat_message('assistant').markdown(explain_prediction(model, r))

    g1, g2 = st.columns([1.5, 1])
    with g1:
        hist = model.demand[model.demand['product'] == prod].tail(60)
        obs = hist[hist['kind'].isin(['남음', '품절', '마감 품절'])]
        sold = np.where(obs['kind'] == '남음', obs['demand'], obs['production'])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=obs['date'], y=sold, name='팔린 개수', marker_color='#C9D3DD'))
        fig.add_trace(go.Scatter(x=hist['date'], y=hist['production'], name='생산량',
                                 mode='lines', line=dict(color='#7F8C8D', dash='dot', shape='hv')))
        cens = obs[obs['censored']]
        fig.add_trace(go.Scatter(x=cens['date'], y=cens['demand'], name='품절 후 복원한 수요',
                                 mode='markers', marker=dict(color='#E74C3C', size=9, symbol='x')))
        pf = plan[plan['product'] == prod]
        fig.add_trace(go.Scatter(x=pf['date'], y=pf['mu'], name='예상 판매량', mode='markers+lines',
                                 marker=dict(color='#2E86C1', size=10, symbol='diamond')))
        fig.add_trace(go.Scatter(x=pf['date'], y=pf['production'], name='권장 생산량', mode='markers',
                                 marker=dict(color='#1E8449', size=11, symbol='triangle-up')))
        fig.update_layout(height=380, hovermode='x unified', margin=dict(t=40, b=30, l=10, r=10),
                          title=f"{prod} - 최근 마감 기록과 예측",
                          legend=dict(orientation='h', yanchor='bottom', y=1.0))
        st.plotly_chart(fig, width='stretch')
    with g2:
        st.dataframe(prediction_chain(model, r), hide_index=True, width='stretch')


def render_steps_tab(model):
    cfg = model.cfg
    d0 = next_open_days(model, 1)[0]
    st.subheader("계산 과정 (단계별)")
    st.caption(f"학습 기준일 {model.origin:%Y-%m-%d}, 예측일 {d0:%Y-%m-%d} 기준")

    # 1
    st.markdown("#### 1. 매입에서 재료 일소비량 계산")
    st.caption("빵집은 재료가 떨어질 즈음 삽니다. 그래서 한 번 산 재료는 다음 매입일까지 고르게 쓴다고 봅니다. "
               "월요일에 밀가루 120kg을 사고 7일 뒤 다시 샀다면 하루 약 17kg을 쓴 셈입니다.")
    if model.usage.empty:
        st.info("매입장부가 없거나, 두 번 이상 산 재료가 없어 이 단계를 건너뛰었습니다.")
    else:
        u = model.usage.loc[:model.origin].tail(120)
        fig = px.line(u, labels={'value': '하루 사용량', 'index': '날짜', 'variable': '재료'},
                      title="재료별 하루 사용량 (최근 120일, 단위는 매입 단위 그대로)")
        fig.update_layout(height=320, margin=dict(t=40, b=30, l=10, r=10), hovermode='x unified')
        st.plotly_chart(fig, width='stretch')

    # 2
    st.markdown("#### 2. 역레시피 생산 추정과 사전값")
    st.caption(f"최근 {cfg.bom_window_days}일 재료 소비량에서 제품별로 하루 몇 개를 만들었는지 거꾸로 계산합니다. "
               f"계산이 불안정할 때는 '평소 생산량' 구성비가 결과를 잡아줍니다(bom_lambda {cfg.bom_lambda}). "
               "이 값이 매일 입력이 없을 때 쓰는 사전값입니다.")
    pr = model.inputs.products
    st.dataframe(pd.DataFrame({
        '평소 생산량': pr['usual_qty'],
        '역레시피 추정 생산량': model.bom.production.round(1),
        f"사전값 (생산량의 {1 - cfg.prior_waste:.0%})": model.levels['prior'].astype(float).round(1),
    }).rename_axis('제품'), width='stretch')
    if model.bom.method == '역레시피':
        fit_df = pd.DataFrame({'실제 사용량': model.bom.usage, '추정 생산량으로 되짚은 사용량': model.bom.fitted})
        fit_df['차이'] = ((fit_df.iloc[:, 1] / fit_df.iloc[:, 0] - 1) * 100).round(1).astype(str) + '%'
        with st.expander("재료별 맞춤 정도 (영업일 하루 기준, 로스 제외)"):
            st.dataframe(fit_df.round(3).rename_axis('재료'), width='stretch')
            st.caption(f"전체 규모는 평소 생산량의 {model.bom.scale:.2f}배로 추정됐습니다.")

    # 3
    st.markdown("#### 3. 진짜 수요 복원")
    share18 = cumulative_share(18, cfg)
    st.caption(
        "똑같이 30개 품절이어도 18시 품절과 21시 품절은 의미가 다릅니다. 시간대별 판매 비중으로 품절 뒤에 숨은 "
        f"수요를 복원합니다. 지금 프로필에서는 18시까지 하루 수요의 {share18:.0%}가 들어오므로 18시 품절이면 "
        f"30 / {share18:.2f} = {30 / share18:.0f}개입니다. 3개가 남았다면 수요는 그대로 27개입니다. "
        f"아주 이른 품절로 과대추정되지 않도록 보정은 생산량의 {cfg.max_uplift:g}배까지만 인정합니다."
    )
    dm = model.demand
    if dm.empty:
        st.info("아직 마감 기록(daily_log.csv)이 없습니다. 사전값만으로 예측합니다.")
    else:
        counts = dm['kind'].value_counts()
        st.write(", ".join(f"{k} {v}건" for k, v in counts.items()))
        show = dm.sort_values('date', ascending=False).head(40).assign(
            날짜=lambda x: x['date'].dt.strftime('%Y-%m-%d'),
            판매비중=lambda x: x['share_at_sellout'].map(lambda v: '' if pd.isna(v) else f"{v:.0%}"),
        )
        st.dataframe(show[['날짜', 'product', 'production', 'result', 'kind', '판매비중', 'demand']]
                     .rename(columns={'product': '제품', 'production': '생산량', 'result': '입력값',
                                      'kind': '해석', '판매비중': '품절 시각까지 판매 비중',
                                      'demand': '복원 수요'}).round(1),
                     hide_index=True, width='stretch', height=300)

    # 4
    st.markdown("#### 4. 수준, 요일 계수 학습")
    st.caption(f"요일 효과를 뺀 뒤 최근 값에 가중치를 더 주는 평균(EWMA, alpha {cfg.alpha})으로 수준을 학습하고, "
               f"관측값과 사전값을 n/(n+k) 비율로 섞습니다 (k={cfg.k_cred:g}: 5일이면 50:50, 20일이면 관측 80%). "
               f"요일 계수도 같은 방식이라 관측이 적은 요일은 매장 전체 평균 쪽에 무게를 둡니다(k={cfg.k_weekday:g}).")
    lv = model.levels
    st.dataframe(pd.DataFrame({
        '사전값': lv['prior'].astype(float).round(1),
        '관측 수준(EWMA)': lv['ewma'].astype(float).round(1),
        '입력 일수 n': lv['n_obs'],
        '품절 복원 일수': lv['n_censored'].astype(int),
        '관측 비중': (lv['weight'].astype(float) * 100).round().astype(int).astype(str) + '%',
        '학습 수준': lv['level'].astype(float).round(1),
        '변동계수(CV)': lv['cv'].astype(float).round(3),
    }).rename_axis('제품'), width='stretch')
    wd = pd.concat([model.store_weekday.to_frame('매장 전체').T, model.weekday])
    wd.columns = WEEKDAY_NAMES
    st.dataframe(wd.round(2).style.background_gradient(cmap='RdBu_r', vmin=0.6, vmax=1.4, axis=None),
                 width='stretch')

    # 5
    st.markdown("#### 5. 시즌, 이벤트 보정")
    if model.season_enabled:
        sf = model.season_factors(d0)
        st.caption(f"매입 이력이 1년 이상이라, 작년 같은 시기에 지금부터 예측일({d0:%m/%d})까지 생산량이 얼마나 "
                   f"늘거나 줄었는지를 매입 흐름으로 계산해 반영합니다 "
                   f"({cfg.season_bounds[0]:g}~{cfg.season_bounds[1]:g}로 제한).")
        st.dataframe(sf.round(3).to_frame('시즌 보정').rename_axis('제품').T, width='stretch')
    else:
        st.caption("매입 이력이 1년 미만이라 시즌 보정은 쓰지 않았습니다.")
    tags = model.event_tags(d0)
    st.caption("이벤트 배수: " + ", ".join(f"{tag_label(t)} {f:g}배" for t, f in cfg.event_factors.items())
               + f". {d0:%m/%d} 이벤트: " + (", ".join(tag_label(t) for t in tags) or "없음"))

    # 6
    st.markdown("#### 6. 권장 생산량")
    st.caption("예측 수요에 마진과 폐기 손실을 고려한 여유분을 더합니다. "
               "목표 서비스수준(품절이 나지 않을 확률) = (판매가 - 원가) / (판매가 - 회수가).")
    raw_sl = (pr['price'] - pr['cost']) / (pr['price'] - pr['salvage']).where(lambda s: s > 0)
    applied = pd.Series({p: model.predict(p, d0)['service_level'] for p in pr.index})
    st.dataframe(pd.DataFrame({
        '판매가': pr['price'], '원가': pr['cost'], '회수가': pr['salvage'],
        '계산값': raw_sl.round(3), '적용 서비스수준': applied.round(3),
    }).rename_axis('제품'), width='stretch')


def parse_whatif_text(text, tags_available):
    """채팅 문구에서 이벤트 태그, 요일을 읽는다. LLM이 없으면 키워드 매칭(NLU 아님)."""
    parsed = parse_events_llm(text)
    if parsed is not None:
        tags = [t for t in ('holiday', 'rain', 'event') if parsed.get(t)]
        weekday = parsed.get('weekday')
    else:
        tags = [t for t, kws in WHATIF_KEYWORDS.items() if any(k in text for k in kws)]
        weekday = next((i for i, n in enumerate(WEEKDAY_NAMES) if f"{n}요일" in text), None)
        if weekday is None and '주말' in text:
            weekday = 5
    return [t for t in tags if t in tags_available], weekday


def render_whatif_tab(model):
    cfg = model.cfg
    st.subheader("What-if - 조건별 예상 판매량")
    st.caption("날짜와 이벤트 조건을 바꾸면 예상 판매량과 권장 생산량이 바로 바뀝니다. "
               "계산은 '생산량 추천'과 같은 모델(요일 계수, 시즌, 이벤트 배수, 목표 서비스수준)입니다.")

    c1, c2 = st.columns([1, 1])
    prod = c1.selectbox("제품", list(model.inputs.products.index), key='wi_product')
    date = pd.Timestamp(c2.date_input(
        "날짜", value=next_open_days(model, 1)[0].date(),
        min_value=(model.origin + pd.Timedelta(days=1)).date(), key='wi_date'))
    if date.weekday() in cfg.closed_weekdays:
        st.warning(f"{WEEKDAY_NAMES[date.weekday()]}요일은 정기 휴무일입니다.")
        return

    st.markdown("##### 이벤트 조건")
    planned = set(model.event_tags(date))
    cols = st.columns(max(len(cfg.event_factors), 1))
    chosen = []
    for col, (tag, factor) in zip(cols, cfg.event_factors.items()):
        if col.checkbox(f"{tag_label(tag)} ({factor:g}배)", value=tag in planned, key=f"wi_tag_{tag}_{date:%Y%m%d}"):
            chosen.append(tag)
    if planned:
        st.caption("events.csv 에 이 날짜로 적힌 이벤트는 미리 체크해 두었습니다.")

    base = model.predict(prod, date, tags=[])
    live = model.predict(prod, date, tags=chosen)
    pct = (live['mu'] / base['mu'] - 1) * 100 if base['mu'] > 0 else 0.0
    r1, r2, r3 = st.columns(3)
    r1.metric(f"조건 없을 때 ({live['weekday']})", f"{base['mu']:.0f}개")
    r2.metric("조건 반영 예상 판매량", f"{live['mu']:.0f}개", delta=f"{pct:+.1f}%")
    r3.metric(f"권장 생산량 (목표 {live['service_level']:.0%})", f"{live['production']}개")

    with st.expander("모든 제품 보기"):
        rows = []
        for p in model.inputs.products.index:
            b, l = model.predict(p, date, tags=[]), model.predict(p, date, tags=chosen)
            rows.append({'제품': p, '조건 없을 때': round(b['mu'], 1), '조건 반영': round(l['mu'], 1),
                         '권장 생산량': l['production'], '평소 생산량': int(l['usual_qty'])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    # ---------- 채팅 ----------
    st.markdown("---")
    st.markdown("##### 문장으로 물어보기")
    st.caption("예: '토요일에 비 오면?', '공휴일에 행사까지 하면 크루아상 몇 개?' "
               " - 요일을 말하면 그 요일의 가장 가까운 날로 계산합니다.")
    if 'whatif_messages' not in st.session_state:
        st.session_state.whatif_messages = []
    if st.button("대화 초기화", key='whatif_clear'):
        st.session_state.whatif_messages = []
        st.rerun()
    for msg in st.session_state.whatif_messages:
        st.chat_message(msg['role']).write(msg['content'])

    user_text = st.chat_input(f"'{prod}'에 대해 물어보세요 (예: 토요일에 비 오면?)")
    if not user_text:
        return
    st.session_state.whatif_messages.append({'role': 'user', 'content': user_text})
    tags, weekday = parse_whatif_text(user_text, set(cfg.event_factors))
    q_date = date
    if weekday is not None and weekday not in cfg.closed_weekdays:
        q_date = next(d for d in next_open_days(model, 14) if d.weekday() == weekday)
    b = model.predict(prod, q_date, tags=[])
    r = model.predict(prod, q_date, tags=tags)
    cond_str = ", ".join(tag_label(t) for t in tags) or "특별한 이벤트 없음"
    pct = (r['mu'] / b['mu'] - 1) * 100 if b['mu'] > 0 else 0.0
    reply = (f"{q_date:%m/%d}({r['weekday']}), 조건({cond_str})이면 '{prod}' 예상 판매량은 "
             f"{b['mu']:.0f}개에서 **{r['mu']:.0f}개**로 바뀝니다({pct:+.1f}%). "
             f"품절이 나지 않을 확률 {r['service_level']:.0%}를 목표로 한 권장 생산량은 **{r['production']}개**입니다.")
    ctx = {'item': prod, 'weekday': f"{r['weekday']}요일", 'cond_str': cond_str, 'base': b['mu'],
           'final': r['mu'], 'pct': pct, 'sl': r['service_level'], 'production': r['production']}

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
    st.session_state.whatif_messages.append({'role': 'assistant', 'content': streamed})


# -----------------------------------------------------------------------------
# 10. 메인 UI 구성
# -----------------------------------------------------------------------------
st.title(f"{CURRENT_USER['shop_name']} - AI 수요 예측")
st.write("매입장부와 매일 마감 기록 한 줄로 내일 몇 개 만들지 추천합니다.")

USER_DIR = auth_store.user_data_dir(CURRENT_USER['id'])
if 'cfg_base' not in st.session_state:
    _cfg_path = USER_DIR / CONFIG_FILE
    st.session_state.cfg_base = (Config.load(_cfg_path) if _cfg_path.exists() else Config()).to_dict()

# --- 사용자별 입력 파일 ---
# 올린 파일은 계정별로 분리된 data/users/<user_id>/ 아래 고정 이름으로 저장된다.
# 다음 접속 때 다시 올릴 필요 없이 마지막 파일을 그대로 쓴다(daily_log 는 매일 새로 올리면 덮어쓴다).
with st.sidebar:
    st.header("입력 파일")
    use_sample = st.checkbox("동봉된 예시 데이터로 둘러보기", key='main_sample')
    blobs = {}
    if use_sample:
        for key, fname in FILE_NAMES.items():
            if (SAMPLE_INPUT_DIR / fname).exists():
                blobs[key] = (SAMPLE_INPUT_DIR / fname).read_bytes()
    else:
        for key, fname in FILE_NAMES.items():
            saved = USER_DIR / fname
            up = st.file_uploader(fname, type=['csv'], key=f'up_{key}', help=INPUT_HELP[key])
            if up is not None:
                data = up.getvalue()
                try:
                    if not saved.exists() or saved.read_bytes() != data:
                        saved.write_bytes(data)
                        auth_store.record_dataset(
                            CURRENT_USER['id'], fname, saved,
                            n_rows=max(len(data.splitlines()) - 1, 0),
                        )
                except Exception as e:
                    st.warning(f"{fname} 보관에 실패했습니다(분석은 그대로 진행됩니다): {e}")
                blobs[key] = data
            elif saved.exists():
                blobs[key] = saved.read_bytes()
        st.caption("  \n".join(
            f"{f}: " + ("저장된 파일 사용" if k in blobs and st.session_state.get(f'up_{k}') is None
                        else "올림" if k in blobs else "없음(선택)" if k == 'events' else "없음")
            for k, f in FILE_NAMES.items()
        ))

    with st.expander("입력 양식 내려받기"):
        for key, fname in FILE_NAMES.items():
            if (SAMPLE_INPUT_DIR / fname).exists():
                st.download_button(fname, (SAMPLE_INPUT_DIR / fname).read_bytes(), file_name=fname,
                                   mime='text/csv', key=f'tpl_{key}', help=INPUT_HELP[key])

tab_plan, tab_steps, tab_whatif, tab_cfg, tab_alert, tab_shap, tab_h1 = st.tabs([
    "생산량 추천", "계산 과정", "What-if 채팅", "환경 변수",
    "판매 이상 알림 (POS)", "SHAP 기여도 (POS)", "이벤트 우선 예측 (H1)",
])

# 설정 탭을 먼저 그려야 바뀐 값으로 예측할 수 있다(화면상 탭 순서와 무관).
with tab_cfg:
    CFG = render_config_editor(Config.from_dict(st.session_state.cfg_base), USER_DIR)

MODEL, LOAD_ERROR = None, None
if 'products' in blobs:
    try:
        with st.spinner("매입장부와 마감 기록을 학습하는 중..."):
            MODEL = build_model(tuple(sorted(blobs.items())),
                                json.dumps(CFG.to_dict(), ensure_ascii=False, sort_keys=True))
        st.sidebar.success(f"학습 완료, 기준일 {MODEL.origin:%Y-%m-%d}")
    except InputError as e:
        LOAD_ERROR = str(e)
    except Exception as e:
        LOAD_ERROR = f"입력 파일을 처리하지 못했습니다: {e}"
if LOAD_ERROR:
    st.sidebar.error(LOAD_ERROR)

for _tab, _render in ((tab_plan, render_plan_tab), (tab_steps, render_steps_tab),
                      (tab_whatif, render_whatif_tab)):
    with _tab:
        if MODEL is None:
            if LOAD_ERROR:
                st.error(LOAD_ERROR)
            render_input_guide()
        else:
            _render(MODEL)

# 탭 3: 판매 이상 알림 (재고 판단용)
with tab_alert:
    st.subheader("판매 이상 알림")
    st.caption(
        "평소와 다르게 팔린 날을 찾아드립니다. "
        "'평소'는 **직전 4주간 같은 요일의 평균**입니다 - 토요일은 토요일끼리 비교합니다."
    )

    alert_file = st.file_uploader(
        "판매 데이터 CSV 업로드 (date, item, sales_qty 컬럼 포함)",
        type=['csv'], key='anomaly_csv'
    )
    if alert_file is None and SAMPLE_DATA_PATH.exists():
        if st.checkbox("동봉된 예시 데이터로 둘러보기", key='anomaly_sample'):
            alert_file = str(SAMPLE_DATA_PATH)

    if alert_file is None:
        st.info("판매 데이터를 올리시면 재고 판단에 쓸 알림을 만들어 드립니다.")
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
                    # 시즌, 공휴일, 날씨를 알면 '왜 그랬는지'까지 말해줄 수 있다.
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
                st.markdown("### 이런 날이 있었어요")
                if not alerts:
                    st.success(
                        f"최근 {period}일 동안 평소와 크게 다른 날이 없었습니다. "
                        "판매가 안정적입니다."
                    )
                else:
                    n_event = sum(1 for a in alerts if a['is_event'])
                    if n_event:
                        st.caption(
                            f"이 가운데 {n_event}건은 시즌, 공휴일, 날씨로 설명되는 날입니다. "
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
                                tag = '이유를 아는 날'
                            else:
                                tag = {'높음': '눈여겨보세요',
                                       '보통': '참고하세요',
                                       '낮음': '가벼운 차이'}[a['severity']]
                            st.markdown(f"**{a['headline']}**")
                            st.caption(f"{tag}, {a['detail']}")
                            st.markdown(f"{a['meaning']} {a['action']}")
                            if a['cause']:
                                st.caption(a['cause'])

                    if len(shown) > 20:
                        st.caption(f"이 밖에 {len(shown) - 20}건이 더 있습니다.")

                # ---------- 2. 요즘 추세 ----------
                st.markdown("### 요즘 잘 나가고 있나요?")
                st.caption("하루 튄 것보다 **몇 주에 걸친 변화**가 발주량 조정에 더 중요합니다.")

                trend_rows = []
                for it, t in trends.items():
                    if t is None:
                        continue
                    trend_rows.append({
                        '품목': it,
                        '방향': ('늘고 있음' if t['direction'] == '상승' else '줄고 있음'),
                        '최근 4주 변화': f"{t['change'] * 100:+.0f}%",
                        '이렇게 해보세요': t['action'],
                    })

                if trend_rows:
                    trend_df = pd.DataFrame(trend_rows).sort_values(
                        '최근 4주 변화',
                        key=lambda s: s.str.rstrip('%').astype(float),
                        ascending=False,
                    )
                    st.dataframe(trend_df, width='stretch', hide_index=True)
                else:
                    st.info("최근 4주간 뚜렷하게 늘거나 줄어든 품목이 없습니다.")

                # ---------- 3. 요일별 준비량 ----------
                st.markdown("### 요일별 준비량 가이드")
                st.caption("최근 4주 실적을 요일별로 모은 표입니다. 내일 몇 개 준비할지 정할 때 쓰세요.")

                guide_item = st.selectbox("품목 선택", all_items, key='guide_item')
                guide = weekday_guide(by_item[guide_item])

                if guide is None:
                    st.info("이 품목은 데이터가 부족해 요일별 가이드를 만들 수 없습니다.")
                else:
                    gc1, gc2 = st.columns([1, 1.4])
                    with gc1:
                        st.dataframe(guide, width='stretch', hide_index=True)
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
                            title=f"{guide_item} - 최근 60일",
                            margin=dict(t=45, b=30, l=10, r=10),
                            legend=dict(orientation='h', yanchor='bottom', y=1.0),
                        )
                        st.plotly_chart(figg, width='stretch')

                # ---------- 설명 ----------
                with st.expander("이 알림은 어떻게 만들어지나요?"):
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
                        "그 날 매장 전체 매출도 흔들렸다면 날씨, 휴일 같은 바깥 요인일 가능성이 크고, "
                        "그 품목만 흔들렸다면 진열, 품질, 경쟁 제품 등 품목 자체의 문제일 가능성이 큽니다."
                    )

# 탭 4: SHAP 기여도 분석
with tab_shap:
    st.subheader("SHAP 기여도 분석 (XAI)")
    st.caption(
        "예측치를 '기저치 + 이벤트별 기여도'로 분해합니다. 각 예측이 왜 그 값이 나왔는지 "
        "품목, 날짜 단위로 근거를 확인할 수 있습니다."
    )
    shap_file = st.file_uploader(
        "이벤트 통합 데이터셋 CSV 업로드 (date, item, sales_qty, 이벤트, 기온 컬럼 포함)",
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
                st.plotly_chart(fig6a, width='stretch')

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
                st.plotly_chart(fig6b, width='stretch')

                with st.expander("기여도 상세 표"):
                    st.dataframe(contrib_df, width='stretch', hide_index=True)
                    st.caption(
                        f"검증: 기저치({base_value:.1f}) + 전체 기여도 합({contribs.sum():.1f}) "
                        f"= 모델 예측치({pred:.1f})"
                    )
    else:
        st.info("이벤트 통합 데이터셋 CSV를 업로드하면 SHAP 기여도 분석을 실행할 수 있습니다.")

# 탭 6: 이벤트 우선 게이팅(H1) 검증 - 개발자 전용, 맨 뒤에 배치
with tab_h1:
    st.error(
        "**개발자 전용 탭입니다.** 모델이 제대로 동작하는지 검증하는 화면이라 "
        "매장 운영에는 필요하지 않습니다. 그냥 지나치셔도 됩니다."
    )
    st.subheader("이벤트 우선 게이팅 (Layer 2, H1) - Walk-forward 백테스트")
    st.caption(
        "공휴일, 방학, 시즌, 날씨 이벤트를 반영한 예측이, 이벤트를 무시한 기저 시계열 예측보다 "
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
                "이벤트 게이팅 임계치 tau (이 값을 넘는 이벤트 강도만 예측에 반영)",
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
                    "참고: '곱셈형 게이팅'은 기존 융합식 y_hat_basex(1+sumbeta_ke_k)를 그대로 둔 **대조군(ablation)**입니다. "
                    "Prophet이 주간 계절성으로 이미 학습한 주말 효과를 beta_weekend(+0.35~+0.43)로 한 번 더 "
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
                    title=f"{ev_item} - 이벤트 게이팅 백테스트 (주황 음영 = 게이팅 발동일)"
                )
                st.plotly_chart(fig3, width='stretch')

                with st.expander("추정된 이벤트 탄력도(beta) 보기"):
                    beta_df = pd.DataFrame({'이벤트': list(beta.keys()), '탄력도(beta)': list(beta.values())})
                    beta_df = beta_df.reindex(
                        beta_df['탄력도(beta)'].abs().sort_values(ascending=False).index
                    )
                    st.dataframe(beta_df, width='stretch', hide_index=True)
                    st.caption("beta > 0: 해당 이벤트일 때 판매량 증가 경향 / beta < 0: 감소 경향 "
                               "(학습 구간 기준 선형회귀 추정치, sumbeta_k, e_k가 이벤트 강도 점수)")
    else:
        st.info("방학, 공휴일, 날씨, 시즌 이벤트가 포함된 통합 데이터셋 CSV를 업로드하면 H1 가설을 검증할 수 있습니다.")
