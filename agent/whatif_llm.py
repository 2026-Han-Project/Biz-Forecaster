"""What-if 질문에서 이벤트 태그, 요일을 추출한다. 실패 시 None -> 호출부가 키워드 매칭으로 폴백."""

import json
from agent.brief_llm import _setup_env, MODEL_NAME

EVENT_KEYS = ["holiday", "rain", "event"]

PARSE_SYSTEM = """사용자의 질문에서 해당되는 상황을 추출하는 파서입니다.

출력 규칙:
- 키는 holiday, rain, event, weekday 4개만 사용합니다.
- holiday(공휴일/명절/연휴/빨간날), rain(비/장마/우천), event(행사/축제/시즌 이벤트)는 true 또는 false입니다.
- weekday는 질문에 요일이 나오면 0(월)~6(일) 정수, 주말이면 5, 없으면 null입니다.
- 질문에 명시되지 않았거나 확실하지 않으면 false 또는 null로 둡니다.
- 부정 표현("비 안 오면")은 false로 처리합니다.
- JSON 객체만 출력하고 설명, 마크다운 기호는 절대 붙이지 않습니다."""


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
        out = {k: bool(parsed.get(k, False)) for k in EVENT_KEYS}
        wd = parsed.get("weekday")
        out["weekday"] = int(wd) if isinstance(wd, (int, float)) and 0 <= int(wd) <= 6 else None
        return out

    except Exception as e:
        print(f"[whatif_llm] 파싱 실패: {e}")
        return None

REPLY_SYSTEM = """당신은 베이커리 사장님에게 What-if 시뮬레이션 결과를 설명하는 분석가입니다.

작성 규칙:
- 한국어 존댓말, 2~3문장
- 주어진 숫자만 사용하며 새로운 수치를 만들지 않습니다
- 조건이 판매량에 어떤 방향으로 작용하는지 먼저 말하고, 권장 생산량으로 마무리합니다
- 권장 생산량이 예상 판매량보다 많은 이유(목표 서비스수준)를 짧게 덧붙입니다
- 마크다운 강조 기호는 사용하지 않습니다"""

REPLY_TEMPLATE = """제품: {item}
요일: {weekday}
적용 조건: {cond_str}
조건 없을 때 예상 판매량: {base:.0f}개
조건 반영 예상 판매량: {final:.0f}개 ({pct:+.1f}%)
목표 서비스수준(품절이 나지 않을 확률): {sl:.0%}
권장 생산량: {production}개
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
