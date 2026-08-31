"""이메일 인증코드 발송 — 네이버 / 구글(Gmail) SMTP.

두 서비스 모두 표준 SMTP를 열어두고 있어서, 별도 SDK 없이 파이썬 표준
라이브러리(smtplib, email)만으로 보낼 수 있다. 발송 계정 정보는 코드가 아니라
.streamlit/secrets.toml에 둔다 (이 파일은 .gitignore에 이미 등록돼 있다).

  .streamlit/secrets.toml 예시
  ---------------------------------------------------------------
  [email]
  provider    = "naver"              # "naver" 또는 "gmail"
  sender      = "myshop@naver.com"   # 보내는 계정
  password    = "발급받은-앱-비밀번호"
  sender_name = "Biz Forecaster"
  ---------------------------------------------------------------

앱 비밀번호를 따로 발급받아야 하는 이유
    두 서비스 다 계정 로그인 비밀번호로는 외부 SMTP 접속을 막아둔다.
    앱 전용 비밀번호는 메일 발송 권한만 갖고 언제든 개별 폐기할 수 있어서,
    유출돼도 계정 전체가 넘어가지 않는다.

  네이버 : 네이버 메일 > 환경설정 > POP3/IMAP 설정 > "IMAP/SMTP 사용" 켜기
           2단계 인증을 쓰면 네이버 계정관리에서 애플리케이션 비밀번호를 발급
  구글   : Google 계정 > 보안 > 2단계 인증 켜기 > 앱 비밀번호 생성(16자리)

설정이 없으면 개발 모드로 동작한다. 메일을 보내지 않고 코드를 화면에 띄워
로그인 흐름 자체는 그대로 시험해볼 수 있다.
"""

import os
import smtplib
import ssl
from email.message import EmailMessage

# provider 이름 -> (SMTP 호스트, 포트, STARTTLS 사용 여부)
SMTP_PROFILES = {
    "naver": ("smtp.naver.com", 587, True),
    "gmail": ("smtp.gmail.com", 587, True),
    "naver_ssl": ("smtp.naver.com", 465, False),   # STARTTLS가 막힌 망을 위한 대안
    "gmail_ssl": ("smtp.gmail.com", 465, False),
}

SETUP_GUIDE = """\
**이메일 발송 설정이 없어 개발 모드로 동작 중입니다.**

실제 메일로 인증코드를 받으려면 `.streamlit/secrets.toml`에 아래를 추가하세요:

```toml
[email]
provider    = "naver"            # 또는 "gmail"
sender      = "본인계정@naver.com"
password    = "발급받은 앱 비밀번호"
sender_name = "Biz Forecaster"
```

- **네이버**: 메일 > 환경설정 > POP3/IMAP 설정에서 `IMAP/SMTP 사용`을 켠 뒤,
  2단계 인증 사용 시 네이버 계정관리에서 애플리케이션 비밀번호를 발급받습니다.
- **구글**: Google 계정 > 보안에서 2단계 인증을 켠 뒤 `앱 비밀번호`(16자리)를 생성합니다.
  일반 로그인 비밀번호로는 접속이 거부됩니다.
"""


def _cfg_from_streamlit_secrets():
    """.streamlit/secrets.toml 또는 배포 플랫폼의 Secrets 입력란에서 읽는다.

    Streamlit Community Cloud는 앱 설정의 Secrets 칸에 붙여넣은 내용을 실행 시점에
    주입해준다. 그래서 secrets.toml을 .gitignore에 넣어 저장소에서 빼두어도
    배포된 앱은 정상적으로 읽는다 — 비밀키를 코드 저장소에 두지 않기 위한 구조다.
    """
    try:
        import streamlit as st
        return st.secrets.get("email", None)
    except Exception:
        # streamlit 런타임 밖이거나 secrets 파일이 없는 경우
        return None


def _cfg_from_env():
    """환경변수에서 읽는다 (Streamlit Cloud가 아닌 배포 환경용).

    Docker·Render·Railway·회사 서버처럼 Secrets 입력란이 없는 곳에서는
    환경변수가 그 자리를 대신한다. 여기서도 설정 값이 코드에 들어가지 않는다.

        BIZ_EMAIL_PROVIDER=gmail
        BIZ_EMAIL_SENDER=myshop@gmail.com
        BIZ_EMAIL_PASSWORD=앱비밀번호16자리
        BIZ_EMAIL_SENDER_NAME=Biz Forecaster
    """
    sender = os.environ.get("BIZ_EMAIL_SENDER", "")
    password = os.environ.get("BIZ_EMAIL_PASSWORD", "")
    if not sender or not password:
        return None
    return {
        "provider": os.environ.get("BIZ_EMAIL_PROVIDER", "naver"),
        "sender": sender,
        "password": password,
        "sender_name": os.environ.get("BIZ_EMAIL_SENDER_NAME", "Biz Forecaster"),
    }


def load_email_config():
    """발송 설정을 읽는다. 없으면 None (개발 모드).

    secrets를 먼저 보고, 없으면 환경변수를 본다. 두 경로를 모두 지원해야
    로컬 실행 · Streamlit Cloud 배포 · 컨테이너 배포가 같은 코드로 동작한다.
    """
    cfg = _cfg_from_streamlit_secrets() or _cfg_from_env()
    if not cfg:
        return None

    sender = str(cfg.get("sender", "")).strip()
    # 구글 앱 비밀번호는 4자리씩 띄어 표시되므로 공백을 제거해 그대로 붙여넣을 수 있게 한다.
    password = str(cfg.get("password", "")).replace(" ", "").strip()
    if not sender or not password:
        return None

    provider = str(cfg.get("provider", "naver")).strip().lower()
    if provider not in SMTP_PROFILES:
        provider = "naver"

    return {
        "provider": provider,
        "sender": sender,
        "password": password,
        "sender_name": str(cfg.get("sender_name", "Biz Forecaster")).strip(),
    }


def is_configured():
    return load_email_config() is not None


def _build_message(cfg, to_email, code, purpose):
    title = "비밀번호 재설정" if purpose == "reset" else "회원가입"
    msg = EmailMessage()
    msg["Subject"] = f"[Biz Forecaster] {title} 인증코드: {code}"
    msg["From"] = f"{cfg['sender_name']} <{cfg['sender']}>"
    msg["To"] = to_email

    msg.set_content(
        f"Biz Forecaster {title} 인증코드는 [ {code} ] 입니다.\n"
        f"10분 안에 입력해주세요.\n\n"
        f"본인이 요청하지 않았다면 이 메일은 무시하셔도 됩니다."
    )
    # HTML 파트를 함께 붙이되, 평문 파트도 남겨 HTML을 막아둔 메일함에서도 읽히게 한다.
    msg.add_alternative(
        f"""\
<html><body style="font-family:'Malgun Gothic',sans-serif;line-height:1.7;color:#222">
  <h2 style="margin-bottom:4px">🥐 Biz Forecaster</h2>
  <p style="color:#666;margin-top:0">소상공인 AI 수요예측 · 이상탐지 플랫폼</p>
  <hr style="border:none;border-top:1px solid #eee">
  <p>{title} 인증코드입니다. 아래 6자리를 입력창에 넣어주세요.</p>
  <p style="font-size:30px;font-weight:bold;letter-spacing:8px;
            background:#f5f7fa;padding:16px 20px;border-radius:8px;display:inline-block">
    {code}
  </p>
  <p style="color:#888;font-size:13px">이 코드는 <b>10분</b> 후 만료됩니다.
     본인이 요청하지 않았다면 이 메일은 무시하셔도 됩니다.</p>
</body></html>""",
        subtype="html",
    )
    return msg


def send_verification_code(to_email, code, purpose="signup", timeout=15):
    """인증코드 메일을 보낸다. 반환: (성공여부, 사용자에게 보여줄 메시지)

    설정이 없으면 메일을 보내지 않고 (False, 개발모드 안내)를 돌려준다.
    호출하는 UI가 그때는 코드를 화면에 직접 띄운다.
    """
    cfg = load_email_config()
    if cfg is None:
        return False, "DEV_MODE"

    host, port, use_starttls = SMTP_PROFILES[cfg["provider"]]
    msg = _build_message(cfg, to_email, code, purpose)

    try:
        if use_starttls:
            with smtplib.SMTP(host, port, timeout=timeout) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.login(cfg["sender"], cfg["password"])
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, port, timeout=timeout,
                                  context=ssl.create_default_context()) as server:
                server.login(cfg["sender"], cfg["password"])
                server.send_message(msg)
        return True, f"{to_email}로 인증코드를 보냈습니다. 메일함을 확인해주세요."

    except smtplib.SMTPAuthenticationError:
        return False, (
            "SMTP 인증에 실패했습니다. 일반 로그인 비밀번호가 아니라 "
            "**앱 비밀번호**를 넣었는지, 네이버는 IMAP/SMTP 사용이 켜져 있는지 확인해주세요."
        )
    except smtplib.SMTPRecipientsRefused:
        return False, "받는 사람 주소가 거부되었습니다. 이메일을 다시 확인해주세요."
    except (smtplib.SMTPException, OSError) as e:
        return False, f"메일 발송에 실패했습니다: {type(e).__name__} — {e}"
