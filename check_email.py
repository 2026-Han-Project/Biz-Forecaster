"""이메일 발송 설정이 제대로 됐는지 확인하는 도구.

앱을 띄우지 않고 SMTP 접속만 따로 시험한다. SMTP 오류 메시지는 원문이 불친절해서
(예: "Username and Password not accepted") 원인을 짚어주지 않으면 헤매기 쉽다.

  사용법
    .venv\\Scripts\\python.exe check_email.py           접속·로그인만 확인 (메일 발송 없음)
    .venv\\Scripts\\python.exe check_email.py --send     실제 테스트 메일 1통 발송

  --send 는 secrets.toml 의 sender 주소, 즉 본인에게만 보낸다.
  실행 중에 한 번 더 확인을 묻는다.
"""

import smtplib
import ssl
import sys
import tomllib
from pathlib import Path

SECRETS = Path(__file__).parent / ".streamlit" / "secrets.toml"

# auth/mailer.py 와 같은 프로필을 쓴다 (한쪽만 고치는 실수를 막기 위해 직접 import)
sys.path.insert(0, str(Path(__file__).parent))
from auth.mailer import SMTP_PROFILES, _build_message, _cfg_from_env  # noqa: E402


def fail(msg, hint=None):
    print(f"\n  [실패] {msg}")
    if hint:
        print(f"\n  {hint}")
    print()
    input("  엔터를 누르면 창이 닫힙니다...")
    sys.exit(1)


def main():
    send_test = "--send" in sys.argv

    print()
    print("=" * 62)
    print("  Biz Forecaster — 이메일 발송 설정 확인")
    print("=" * 62)

    # --- 1. 설정 읽기 --------------------------------------------------------
    # 앱과 같은 순서로 본다: secrets.toml 먼저, 없으면 환경변수.
    cfg, source = None, None

    if SECRETS.exists():
        try:
            cfg_all = tomllib.loads(SECRETS.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            fail(f"secrets.toml 형식이 잘못됐습니다: {e}",
                 "따옴표나 대괄호가 빠지지 않았는지 확인해주세요.")
        if cfg_all.get("email"):
            cfg, source = cfg_all["email"], f"secrets.toml ({SECRETS})"

    if cfg is None:
        env_cfg = _cfg_from_env()
        if env_cfg:
            cfg, source = env_cfg, "환경변수 (BIZ_EMAIL_*)"

    if cfg is None:
        fail(
            "이메일 설정을 찾지 못했습니다.",
            "둘 중 하나로 설정하세요:\n"
            f"    1) {SECRETS} 파일에 [email] 블록 추가\n"
            "       (secrets.toml.example 을 복사해 쓰면 됩니다)\n"
            "    2) 환경변수 BIZ_EMAIL_SENDER / BIZ_EMAIL_PASSWORD 설정\n"
            "       (Docker·Render 등 배포 환경용)",
        )

    print(f"\n  설정 출처   : {source}")

    provider = str(cfg.get("provider", "")).strip().lower()
    sender = str(cfg.get("sender", "")).strip()
    # 구글 앱 비밀번호는 4자리씩 띄어서 표시된다. 그대로 붙여넣어도 되게 공백을 제거한다.
    password = str(cfg.get("password", "")).replace(" ", "").strip()
    sender_name = str(cfg.get("sender_name", "Biz Forecaster")).strip()

    print(f"\n  보내는 계정 : {sender or '(비어 있음)'}")
    print(f"  제공자      : {provider or '(비어 있음)'}")
    print(f"  비밀번호    : {'*' * len(password) + f'  ({len(password)}자)' if password else '(비어 있음)'}")

    if not sender:
        fail("sender 가 비어 있습니다.")
    if not password:
        fail(
            "password 가 비어 있어 지금은 개발 모드로 동작합니다.",
            "구글 앱 비밀번호 발급:\n"
            "    1) https://myaccount.google.com/security 에서 2단계 인증 켜기\n"
            "    2) https://myaccount.google.com/apppasswords 에서 16자리 생성\n"
            "    3) secrets.toml 의 password = \"\" 안에 붙여넣기\n\n"
            "  (설정하지 않아도 인증코드가 화면에 표시되므로 가입은 가능합니다)",
        )
    if provider not in SMTP_PROFILES:
        fail(f"provider 값이 올바르지 않습니다: {provider!r}",
             f"사용 가능한 값: {', '.join(SMTP_PROFILES)}")

    if provider.startswith("gmail") and len(password) != 16:
        print(f"\n  [주의] 구글 앱 비밀번호는 보통 16자리인데 {len(password)}자입니다.")
        print("         일반 로그인 비밀번호를 넣으신 게 아닌지 확인해주세요.")

    host, port, use_starttls = SMTP_PROFILES[provider]
    print(f"  서버        : {host}:{port} ({'STARTTLS' if use_starttls else 'SSL'})")

    # --- 2. 접속 · 로그인 ----------------------------------------------------
    print("\n  접속을 시도합니다...")
    try:
        if use_starttls:
            server = smtplib.SMTP(host, port, timeout=20)
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
        else:
            server = smtplib.SMTP_SSL(host, port, timeout=20,
                                      context=ssl.create_default_context())
    except (OSError, smtplib.SMTPException) as e:
        fail(
            f"메일 서버에 접속하지 못했습니다: {type(e).__name__} — {e}",
            "회사·학교 네트워크나 백신이 587 포트를 막고 있을 수 있습니다.\n"
            f"  secrets.toml 의 provider 를 \"{provider}_ssl\" 로 바꿔 465 포트로 시도해보세요."
            if not provider.endswith("_ssl") else "네트워크 연결을 확인해주세요.",
        )

    print("  접속 성공. 로그인을 시도합니다...")
    try:
        server.login(sender, password)
    except smtplib.SMTPAuthenticationError as e:
        server.quit()
        fail(
            f"로그인이 거부되었습니다. (서버 응답: {e.smtp_code})",
            "가장 흔한 원인 순서대로:\n"
            "    1) 일반 로그인 비밀번호를 넣은 경우 — 반드시 '앱 비밀번호'여야 합니다\n"
            "    2) 2단계 인증이 꺼져 있는 경우 — 켜야 앱 비밀번호를 만들 수 있습니다\n"
            "    3) sender 주소와 앱 비밀번호를 만든 계정이 다른 경우\n"
            "    4) (네이버) 메일 환경설정에서 IMAP/SMTP 사용이 꺼져 있는 경우",
        )
    except smtplib.SMTPException as e:
        server.quit()
        fail(f"로그인 중 오류: {type(e).__name__} — {e}")

    print("\n  ✅ 로그인 성공 — 이메일 발송 설정이 정상입니다.")

    # --- 3. 실제 발송 (선택) -------------------------------------------------
    if not send_test:
        server.quit()
        print("\n  실제로 메일이 도착하는지까지 보려면:")
        print("      .venv\\Scripts\\python.exe check_email.py --send")
        print()
        input("  엔터를 누르면 창이 닫힙니다...")
        return

    print(f"\n  {sender} (본인) 에게 테스트 메일 1통을 보냅니다.")
    answer = input("  보낼까요? (y/n) ").strip().lower()
    if answer not in ("y", "yes"):
        server.quit()
        print("\n  발송을 취소했습니다.\n")
        input("  엔터를 누르면 창이 닫힙니다...")
        return

    try:
        msg = _build_message(
            {"sender": sender, "sender_name": sender_name}, sender, "123456", "signup"
        )
        server.send_message(msg)
        server.quit()
    except smtplib.SMTPException as e:
        fail(f"발송에 실패했습니다: {type(e).__name__} — {e}")

    print(f"\n  ✅ 발송 완료 — {sender} 메일함을 확인해주세요.")
    print("     (인증코드 123456 짜리 테스트 메일입니다)")
    print("     스팸함에 들어갔다면 '스팸 아님'으로 표시해두세요.")
    print()
    input("  엔터를 누르면 창이 닫힙니다...")


if __name__ == "__main__":
    main()
