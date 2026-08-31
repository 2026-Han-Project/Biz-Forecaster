"""Streamlit 로그인 게이트 UI — 회원가입 / 로그인 / 비밀번호 재설정.

app.py 맨 앞에서 require_login()을 부르면, 로그인하지 않은 방문자에게는
인증 화면만 보이고 그 아래 분석 화면은 아예 그려지지 않는다.

Streamlit 특유의 주의점
    위젯을 건드릴 때마다 스크립트 전체가 처음부터 다시 실행된다.
    그래서 "인증코드를 보냈다" 같은 진행 상태를 지역 변수에 두면 다음 순간 사라진다.
    st.session_state에 담아야 재실행을 건너뛰고 살아남는다.
"""

import streamlit as st

from auth import store
from auth.mailer import SETUP_GUIDE, is_configured, send_verification_code

SESSION_USER_KEY = "auth_user"


# -----------------------------------------------------------------------------
# 세션 헬퍼
# -----------------------------------------------------------------------------

def current_user():
    return st.session_state.get(SESSION_USER_KEY)


def _flash(msg):
    """다음 재실행 때 한 번만 보여줄 메시지를 남긴다.

    st.rerun()을 부르면 화면이 처음부터 다시 그려지면서 방금 띄운 st.success가 사라진다.
    그래서 메시지를 세션에 맡겨두고, 다시 그려진 화면에서 꺼내 보여준 뒤 지운다."""
    st.session_state["auth_flash"] = msg


def _drain_flash():
    msg = st.session_state.pop("auth_flash", None)
    if msg:
        st.success(msg)


def logout():
    for key in (SESSION_USER_KEY, "signup_pending", "reset_pending", "dev_code"):
        st.session_state.pop(key, None)
    st.rerun()


def _deliver_code(email, purpose, pending_key):
    """인증코드를 발급하고 메일로 보낸다. 설정이 없으면 화면 표시용으로 넘긴다."""
    ok, code_or_msg = store.issue_code(email, purpose=purpose)
    if not ok:
        st.error(code_or_msg)          # 쿨다운 등
        return False

    sent, msg = send_verification_code(email, code_or_msg, purpose=purpose)
    if sent:
        st.session_state[pending_key] = email
        st.success(msg)
        return True

    if msg == "DEV_MODE":
        # 발송 설정이 없는 상태. 흐름을 막지 않고 코드를 화면에 띄운다.
        st.session_state[pending_key] = email
        st.session_state["dev_code"] = code_or_msg
        return True

    st.error(msg)
    return False


def _show_dev_code():
    code = st.session_state.get("dev_code")
    if not code:
        return
    st.warning(f"개발 모드 — 인증코드: **{code}**  (실제 메일은 발송되지 않았습니다)")
    with st.expander("실제 메일로 받으려면"):
        st.markdown(SETUP_GUIDE)


# -----------------------------------------------------------------------------
# 탭 1 — 로그인
# -----------------------------------------------------------------------------

def _login_tab():
    _drain_flash()
    with st.form("login_form"):
        email = st.text_input("이메일", key="login_email", placeholder="myshop@gmail.com")
        password = st.text_input("비밀번호", type="password", key="login_pw")
        submitted = st.form_submit_button("로그인", use_container_width=True, type="primary")

    if submitted:
        ok, result = store.verify_login(email, password)
        if ok:
            st.session_state[SESSION_USER_KEY] = result
            st.rerun()
        else:
            st.error(result)


# -----------------------------------------------------------------------------
# 탭 2 — 회원가입 (이메일 인증 2단계)
# -----------------------------------------------------------------------------

def _signup_tab():
    pending = st.session_state.get("signup_pending")

    if not pending:
        st.caption("1단계 — 매장 정보를 입력하면 이메일로 6자리 인증코드를 보내드립니다.")

        notice = store.supported_domain_notice()
        if notice:
            st.info(notice)

        # 이메일 칸만 form 밖에 둔다.
        # st.form 안의 위젯은 제출 버튼을 눌러야 값이 반영되므로, 다른 도메인을 적었을 때
        # 바로 알려주려면 form 밖이어야 한다.
        shop = st.text_input("매장 이름", placeholder="예) 행복한 빵집 신촌점", key="su_shop")
        email = st.text_input("이메일", placeholder="myshop@gmail.com", key="su_email")

        domain_ok = True
        if email and "@" in email:
            domain_ok, domain_msg = store.check_supported_domain(email)
            if not domain_ok:
                st.warning(f"⚠️ {domain_msg}")

        with st.form("signup_form"):
            c1, c2 = st.columns(2)
            pw1 = c1.text_input("비밀번호", type="password", help="8자 이상, 영문+숫자 조합")
            pw2 = c2.text_input("비밀번호 확인", type="password")
            agree = st.checkbox("업로드한 매출 데이터가 예측·이상탐지 분석에 사용되는 것에 동의합니다.")
            # 지원하지 않는 도메인이면 버튼 자체를 막는다.
            # 눌러본 뒤 거절당하는 것보다, 애초에 못 누르게 하는 편이 덜 답답하다.
            submitted = st.form_submit_button(
                "인증코드 받기", use_container_width=True, type="primary",
                disabled=not domain_ok,
            )

        if submitted:
            ok, email_or_msg = store.validate_email(email)
            if not ok:
                st.error(email_or_msg); return
            email = email_or_msg

            ok, msg = store.check_supported_domain(email)
            if not ok:
                st.error(msg); return

            if not (shop or "").strip():
                st.error("매장 이름을 입력해주세요."); return
            if pw1 != pw2:
                st.error("두 비밀번호가 일치하지 않습니다."); return
            ok, msg = store.validate_password(pw1)
            if not ok:
                st.error(msg); return
            if store.email_exists(email):
                st.error("이미 가입된 이메일입니다. 로그인 탭을 이용해주세요."); return
            if not agree:
                st.error("데이터 이용 동의가 필요합니다."); return

            # 코드 검증까지 통과해야 계정을 만든다. 그 전까지는 세션에만 들고 있는다.
            st.session_state["signup_draft"] = {"email": email, "shop": shop.strip(), "pw": pw1}
            if _deliver_code(email, "signup", "signup_pending"):
                st.rerun()
        return

    # 2단계 — 코드 입력
    st.caption(f"2단계 — **{pending}** 로 보낸 6자리 코드를 입력해주세요. (유효시간 10분)")
    _show_dev_code()

    with st.form("signup_verify_form"):
        code = st.text_input("인증코드", max_chars=6, placeholder="000000")
        c1, c2 = st.columns([2, 1])
        done = c1.form_submit_button("가입 완료", use_container_width=True, type="primary")
        resend = c2.form_submit_button("코드 재발송", use_container_width=True)

    if resend:
        _deliver_code(pending, "signup", "signup_pending")
        st.rerun()

    if done:
        ok, msg = store.check_code(pending, code)
        if not ok:
            st.error(msg); return

        draft = st.session_state.get("signup_draft", {})
        ok, msg = store.create_user(
            draft.get("email", pending), draft.get("pw", ""), draft.get("shop", ""),
            verified=True,          # 코드 검증을 통과했으므로 인증 완료 상태로 생성
        )
        if not ok:
            st.error(msg); return

        for key in ("signup_pending", "signup_draft", "dev_code"):
            st.session_state.pop(key, None)
        _flash("가입이 완료되었습니다. 로그인 탭에서 로그인해주세요.")
        st.balloons()
        st.rerun()

    if st.button("← 처음부터 다시 입력"):
        for key in ("signup_pending", "signup_draft", "dev_code"):
            st.session_state.pop(key, None)
        st.rerun()


# -----------------------------------------------------------------------------
# 탭 3 — 비밀번호 재설정
# -----------------------------------------------------------------------------

def _reset_tab():
    pending = st.session_state.get("reset_pending")

    if not pending:
        notice = store.supported_domain_notice()
        if notice:
            st.info(notice)
        with st.form("reset_request_form"):
            email = st.text_input("가입한 이메일", placeholder="myshop@gmail.com")
            submitted = st.form_submit_button("인증코드 받기", use_container_width=True)

        if submitted:
            ok, email_or_msg = store.validate_email(email)
            if not ok:
                st.error(email_or_msg); return
            email = email_or_msg

            ok, msg = store.check_supported_domain(email)
            if not ok:
                st.error(msg); return

            # 가입 여부와 무관하게 같은 안내를 준다.
            # 여기서 "없는 계정입니다"라고 알려주면 가입자 명단을 캐낼 수 있게 된다.
            if store.email_exists(email):
                _deliver_code(email, "reset", "reset_pending")
            else:
                st.session_state["reset_pending"] = email
            st.info("가입된 계정이라면 인증코드를 보냈습니다.")
            st.rerun()
        return

    st.caption(f"**{pending}** 로 보낸 코드를 입력하고 새 비밀번호를 설정하세요.")
    _show_dev_code()

    with st.form("reset_confirm_form"):
        code = st.text_input("인증코드", max_chars=6, placeholder="000000")
        c1, c2 = st.columns(2)
        pw1 = c1.text_input("새 비밀번호", type="password")
        pw2 = c2.text_input("새 비밀번호 확인", type="password")
        submitted = st.form_submit_button("비밀번호 변경", use_container_width=True, type="primary")

    if submitted:
        if pw1 != pw2:
            st.error("두 비밀번호가 일치하지 않습니다."); return
        ok, msg = store.check_code(pending, code)
        if not ok:
            st.error(msg); return
        ok, msg = store.set_password(pending, pw1)
        if not ok:
            st.error(msg); return
        for key in ("reset_pending", "dev_code"):
            st.session_state.pop(key, None)
        _flash("비밀번호가 변경되었습니다. 로그인 탭에서 로그인해주세요.")
        st.rerun()

    if st.button("← 취소"):
        for key in ("reset_pending", "dev_code"):
            st.session_state.pop(key, None)
        st.rerun()


# -----------------------------------------------------------------------------
# 공개 API
# -----------------------------------------------------------------------------

def render_sidebar_account():
    """로그인한 사용자의 계정 패널을 사이드바에 그린다."""
    user = current_user()
    if not user:
        return
    with st.sidebar:
        st.markdown(f"### 🏪 {user['shop_name']}")
        st.caption(user["email"])
        c1, c2 = st.columns(2)
        if c1.button("계정 설정", use_container_width=True):
            open_account_page()
            st.rerun()
        if c2.button("로그아웃", use_container_width=True):
            logout()
        st.divider()


def require_login():
    """로그인 게이트. 로그인 상태면 사용자 dict를, 아니면 인증 화면을 그리고 None을 반환한다."""
    store.init_db()

    user = current_user()
    if user:
        return user

    st.title("🥐 Biz Forecaster")
    st.caption("소상공인 AI 수요예측 · 이상탐지 플랫폼 — 로그인 후 내 매장 데이터를 분석하세요.")

    if not is_configured():
        st.info(
            "현재 이메일 발송이 설정되어 있지 않아 **개발 모드**로 동작합니다. "
            "인증코드가 메일 대신 화면에 표시되므로 지금 바로 가입을 시험해볼 수 있습니다."
        )

    tab_login, tab_signup, tab_reset = st.tabs(["로그인", "회원가입", "비밀번호 찾기"])
    with tab_login:
        _login_tab()
    with tab_signup:
        _signup_tab()
    with tab_reset:
        _reset_tab()

    st.divider()
    st.caption(
        "업로드한 매출 데이터는 계정별로 분리된 폴더에 저장되며, 다른 사용자에게 노출되지 않습니다."
    )
    return None


# -----------------------------------------------------------------------------
# 계정 설정 화면 — 매장명 / 비밀번호 / 이메일 변경, 회원 탈퇴
# -----------------------------------------------------------------------------

ACCOUNT_VIEW_KEY = "show_account"
DELETE_CONFIRM_PHRASE = "탈퇴합니다"


def open_account_page():
    st.session_state[ACCOUNT_VIEW_KEY] = True


def close_account_page():
    for key in (ACCOUNT_VIEW_KEY, "email_change_pending", "dev_code"):
        st.session_state.pop(key, None)


def _refresh_session_user(user_id):
    """DB에서 다시 읽어 세션을 갱신한다. 이걸 빼먹으면 화면에 옛 매장명이 계속 남는다."""
    fresh = store.get_user(user_id)
    if fresh:
        st.session_state[SESSION_USER_KEY] = fresh
    return fresh


def _section_shop_name(user):
    st.subheader("매장 이름")
    with st.form("shop_name_form"):
        name = st.text_input("매장 이름", value=user["shop_name"], max_chars=60)
        if st.form_submit_button("변경하기", type="primary"):
            ok, msg = store.update_shop_name(user["id"], name)
            if ok:
                _refresh_session_user(user["id"])
                _flash(msg)
                st.rerun()
            st.error(msg)


def _section_password(user):
    st.subheader("비밀번호 변경")
    with st.form("pw_change_form"):
        cur = st.text_input("현재 비밀번호", type="password")
        c1, c2 = st.columns(2)
        new1 = c1.text_input("새 비밀번호", type="password", help="8자 이상, 영문+숫자 조합")
        new2 = c2.text_input("새 비밀번호 확인", type="password")
        if st.form_submit_button("변경하기", type="primary"):
            if new1 != new2:
                st.error("새 비밀번호가 서로 다릅니다."); return
            ok, msg = store.change_password(user["id"], cur, new1)
            if ok:
                _flash(msg + " 다음 로그인부터 새 비밀번호를 사용하세요.")
                st.rerun()
            st.error(msg)


def _section_email(user):
    st.subheader("이메일 (로그인 아이디)")
    st.caption(f"현재 아이디: **{user['email']}**")

    pending = st.session_state.get("email_change_pending")

    if not pending:
        st.write("새 이메일로 인증코드를 보내 본인 확인을 거친 뒤 변경합니다.")
        notice = store.supported_domain_notice()
        if notice:
            st.info(notice)
        with st.form("email_change_form"):
            new_email = st.text_input("새 이메일", placeholder="newshop@naver.com")
            pw = st.text_input("현재 비밀번호", type="password")
            if st.form_submit_button("인증코드 받기", type="primary"):
                ok, email_or_msg = store.validate_email(new_email)
                if not ok:
                    st.error(email_or_msg); return
                new_email = email_or_msg
                if new_email == user["email"]:
                    st.error("현재 사용 중인 이메일과 같습니다."); return
                ok, msg = store.check_supported_domain(new_email)
                if not ok:
                    st.error(msg); return
                if store.email_exists(new_email):
                    st.error("이미 다른 계정이 사용 중인 이메일입니다."); return
                if not store.verify_password_by_id(user["id"], pw):
                    st.error("비밀번호가 올바르지 않습니다."); return
                if _deliver_code(new_email, "change_email", "email_change_pending"):
                    st.rerun()
        return

    st.info(f"**{pending}** 로 인증코드를 보냈습니다. 아래에 입력해주세요.")
    _show_dev_code()
    with st.form("email_verify_form"):
        code = st.text_input("인증코드", max_chars=6, placeholder="000000")
        c1, c2 = st.columns([2, 1])
        confirm = c1.form_submit_button("이메일 변경", type="primary", use_container_width=True)
        cancel = c2.form_submit_button("취소", use_container_width=True)

    if cancel:
        st.session_state.pop("email_change_pending", None)
        st.session_state.pop("dev_code", None)
        st.rerun()

    if confirm:
        ok, msg = store.check_code(pending, code)
        if not ok:
            st.error(msg); return
        ok, msg = store.change_email(user["id"], pending)
        if not ok:
            st.error(msg); return
        st.session_state.pop("email_change_pending", None)
        st.session_state.pop("dev_code", None)
        _refresh_session_user(user["id"])
        _flash(f"{msg} 새 아이디: {pending}")
        st.rerun()


def _section_delete(user):
    st.subheader("회원 탈퇴")
    st.warning(
        "탈퇴하면 계정과 **업로드한 매출 데이터가 모두 삭제됩니다.** 되돌릴 수 없습니다."
    )
    with st.expander("탈퇴하기"):
        with st.form("delete_form"):
            pw = st.text_input("비밀번호", type="password")
            phrase = st.text_input(
                f"확인을 위해 '{DELETE_CONFIRM_PHRASE}' 를 입력해주세요",
                placeholder=DELETE_CONFIRM_PHRASE,
            )
            if st.form_submit_button("계정 영구 삭제", type="primary"):
                if phrase.strip() != DELETE_CONFIRM_PHRASE:
                    st.error(f"'{DELETE_CONFIRM_PHRASE}' 를 정확히 입력해주세요."); return
                ok, msg = store.delete_user(user["id"], pw)
                if not ok:
                    st.error(msg); return
                close_account_page()
                st.session_state.pop(SESSION_USER_KEY, None)
                _flash(msg)
                st.rerun()


def render_account_page():
    """계정 설정 전체 화면. 로그인 상태에서만 호출된다."""
    user = current_user()
    if not user:
        return

    if st.button("← 분석 화면으로 돌아가기"):
        close_account_page()
        st.rerun()

    st.title("⚙️ 계정 설정")
    st.caption(f"{user['shop_name']} · {user['email']} · 가입일 {user['created_at'][:10]}")
    _drain_flash()
    st.divider()

    _section_shop_name(user)
    st.divider()
    _section_password(user)
    st.divider()
    _section_email(user)
    st.divider()
    _section_delete(user)
