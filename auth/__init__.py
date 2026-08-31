"""소상공인 계정 인증 패키지 — 회원가입 / 로그인 / 이메일 인증.

  store.py  : SQLite 계정 저장소 + PBKDF2 해싱 + 인증코드 발급/검증
  mailer.py : 네이버 / 구글(Gmail) SMTP 인증코드 발송
  ui.py     : Streamlit 로그인 게이트 화면
"""
