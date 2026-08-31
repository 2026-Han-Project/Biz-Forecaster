import os
import streamlit as st

from agent.prompts import BRIEF_SYSTEM, BRIEF_USER_TEMPLATE

MODEL_NAME = "gpt-4o-mini"


def _setup_env() -> bool:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return False
    os.environ["OPENAI_API_KEY"] = api_key

    # LangSmith(추적 도구) — 키가 있을 때만 활성화
    ls_key = st.secrets.get("LANGSMITH_API_KEY", "")
    if ls_key:
        os.environ["LANGSMITH_API_KEY"] = ls_key
        os.environ["LANGSMITH_TRACING"] = st.secrets.get("LANGSMITH_TRACING", "true")
        os.environ["LANGSMITH_PROJECT"] = st.secrets.get("LANGSMITH_PROJECT", "biz-forecaster")
    return True


def generate_briefing(state: dict) -> str | None:
    try:
        if not _setup_env():
            return None

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3, timeout=20)

        user_msg = BRIEF_USER_TEMPLATE.format(
            item=state.get("item", ""),
            tier=state.get("tier", ""),
            recent_avg=state.get("recent_avg", 0),
            next_avg=state.get("next_avg", 0),
            pct_change=state.get("pct_change", 0),
            trend=state.get("trend", ""),
            peak_week=state.get("peak_week", ""),
            forecast=state.get("forecast", []),
            shap_top=state.get("shap_top", "정보 없음"),
        )

        resp = llm.invoke([
            {"role": "system", "content": BRIEF_SYSTEM},
            {"role": "user", "content": user_msg},
        ])
        text = (resp.content or "").strip()
        return text if text else None

    except Exception as e:
        print(f"[brief_llm] LLM 호출 실패: {e}")
        return None