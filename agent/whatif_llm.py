"""What-if 질문에서 이벤트 플래그를 추출한다. 실패 시 None → 호출부가 키워드 매칭으로 폴백."""

import json
from agent.brief_llm import _setup_env, MODEL_NAME

EVENT_KEYS = [
    "is_weekend", "is_holiday", "is_vacation",
    "is_rain", "is_snow",
    "is_christmas", "is_suneung", "is_valentine", "is_chuseok",
]

PARSE_SYSTEM = """사용자의 질문에서 해당되는 상황 플래그를 추출하는 파서입니다.

출력 규칙:
- 아래 9개 키만 사용하며, 각 값은 true 또는 false입니다.
- is_weekend(주말/토요일/일요일), is_holiday(공휴일/명절/빨간날),
  is_vacation(방학), is_rain(비/장마/우천), is_snow(눈/폭설),
  is_christmas(크리스마스), is_suneung(수능), is_valentine(밸런타인),
  is_chuseok(추석)
- 질문에 명시되지 않았거나 확실하지 않으면 false로 둡니다.
- 부정 표현("비 안 오면")은 false로 처리합니다.
- JSON 객체만 출력하고 설명·마크다운 기호는 절대 붙이지 않습니다."""


def parse_events_llm(user_text: str) -> dict | None:
    try:
        if not _setup_env():
            return None

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=MODEL_NAME, temperature=0, timeout=15)
        resp = llm.invoke([
            {"role": "system", "content": PARSE_SYSTEM},
            {"role": "user", "content": user_text},
        ])

        raw = (resp.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        # 누락 키 보정 + 예상 밖 키 제거
        return {k: bool(parsed.get(k, False)) for k in EVENT_KEYS}

    except Exception as e:
        print(f"[whatif_llm] 파싱 실패: {e}")
        return None

REPLY_SYSTEM = """당신은 베이커리 사장님에게 What-if 시뮬레이션 결과를 설명하는 분석가입니다.

작성 규칙:
- 한국어 존댓말, 2~3문장
- 주어진 숫자만 사용하며 새로운 수치를 만들지 않습니다
- 조건이 판매량에 어떤 방향으로 작용하는지 먼저 말하고, 권장 생산량으로 마무리합니다
- 게이팅 미발동이라고 표시된 경우 그 사실을 짧게 덧붙입니다
- 마크다운 강조 기호는 사용하지 않습니다"""

REPLY_TEMPLATE = """품목: {item}
적용 조건: {cond_str}
기저 예측: {base:.0f}개
최종 예측: {final:.0f}개 ({pct:+.1f}%)
권장 생산량(안전재고 10% 포함): {production}개
이벤트 강도: {score:+.2f} / 임계치: {tau:.2f}
게이팅: {gate}
"""


def stream_reply(ctx: dict):
    """LLM 응답을 조각(chunk) 단위로 내보낸다. 실패 시 아무것도 내보내지 않는다."""
    if not _setup_env():
        return

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3, timeout=20)
    user_msg = REPLY_TEMPLATE.format(**ctx)

    for chunk in llm.stream([
        {"role": "system", "content": REPLY_SYSTEM},
        {"role": "user", "content": user_msg},
    ]):
        if chunk.content:
            yield chunk.content