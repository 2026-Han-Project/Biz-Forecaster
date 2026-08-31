"""사용자 계정 저장소 — SQLite + PBKDF2 비밀번호 해싱.

소상공인 각자가 로그인해 자기 매장 데이터만 보도록 하는 게 목적이다.
외부 라이브러리 없이 파이썬 표준 라이브러리(sqlite3, hashlib, secrets)만 쓴다.
requirements.txt를 건드리지 않아도 되고, 이미 설치된 venv에서 바로 돈다.

비밀번호 저장 원칙
    평문 저장은 절대 하지 않는다. PBKDF2-HMAC-SHA256으로 계정마다 다른 salt를
    붙여 20만 번 반복 해싱한다. salt가 계정마다 다르므로 DB가 통째로 유출돼도
    레인보우 테이블로 한꺼번에 뚫리지 않고, 반복 횟수가 크므로 무차별 대입이 느려진다.
    비교는 hmac.compare_digest로 해서 실행시간 차이로 정답을 알아내는
    타이밍 공격을 막는다.

인증코드도 같은 방식으로 해싱해 저장한다. DB를 열어봐도 코드를 알 수 없다.
"""

import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "users.db"

PBKDF2_ITERATIONS = 200_000
CODE_TTL_MINUTES = 10          # 인증코드 유효시간
MAX_CODE_ATTEMPTS = 5          # 코드 입력 시도 제한 (무차별 대입 방지)
CODE_RESEND_COOLDOWN_SEC = 60  # 재발송 쿨다운


# -----------------------------------------------------------------------------
# 연결 · 스키마
# -----------------------------------------------------------------------------

@contextmanager
def _conn(db_path=None):
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=None):
    """테이블이 없으면 만든다. 앱 시작 때마다 호출해도 안전하다."""
    with _conn(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT    NOT NULL UNIQUE,
                shop_name     TEXT    NOT NULL,
                password_hash TEXT    NOT NULL,
                salt          TEXT    NOT NULL,
                is_verified   INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT    NOT NULL,
                last_login_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                email      TEXT    PRIMARY KEY,
                code_hash  TEXT    NOT NULL,
                salt       TEXT    NOT NULL,
                purpose    TEXT    NOT NULL DEFAULT 'signup',
                expires_at TEXT    NOT NULL,
                attempts   INTEGER NOT NULL DEFAULT 0,
                sent_at    TEXT    NOT NULL
            )
        """)
        # 사용자별 업로드 이력 — 어떤 파일을 언제 올렸는지 기록한다.
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_datasets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                filename    TEXT    NOT NULL,
                stored_path TEXT    NOT NULL,
                n_rows      INTEGER,
                uploaded_at TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)


# -----------------------------------------------------------------------------
# 해싱 유틸
# -----------------------------------------------------------------------------

def _hash(value, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", value.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()


def _new_salt():
    return secrets.token_hex(16)


def _now():
    return datetime.now()


def _iso(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------------------------------------------------------
# 입력 검증
# -----------------------------------------------------------------------------

# 인증 메일을 보낼 수 있는 도메인.
# 기술적으로는 어느 도메인이든 받을 수 있지만, 개인 Gmail 계정에서 보낸 메일이
# 다른 메일 서비스(특히 네이버)의 스팸함으로 들어가는 일이 잦다. 코드를 못 받은
# 사용자가 가입에서 막히는 것보다, 처음부터 지원 범위를 밝히는 편이 낫다고 보고 제한했다.
#
# 나중에 열어주려면 이 줄만 고치면 된다:
#   ("gmail.com", "naver.com")   -> 네이버도 허용
#   ()                            -> 모든 도메인 허용
ALLOWED_EMAIL_DOMAINS = ("gmail.com",)


def check_supported_domain(email):
    """가입 가능한 메일 도메인인지 본다. 반환: (지원여부, 안내문)"""
    domain = (email or "").strip().lower().rpartition("@")[2]
    if not ALLOWED_EMAIL_DOMAINS or domain in ALLOWED_EMAIL_DOMAINS:
        return True, ""
    allowed = " 또는 ".join("@" + d for d in ALLOWED_EMAIL_DOMAINS)
    return False, f"{allowed} 계정만 확인이 가능합니다."


def supported_domain_notice():
    """화면에 상시 띄울 안내 문구."""
    if not ALLOWED_EMAIL_DOMAINS:
        return ""
    allowed = " 또는 ".join("@" + d for d in ALLOWED_EMAIL_DOMAINS)
    return f"현재 이메일 인증은 {allowed} 계정만 지원합니다."


def validate_email(email):
    """가벼운 형식 검사. 진짜 존재 여부는 인증코드 수신으로 확인한다."""
    email = (email or "").strip().lower()
    if not email or email.count("@") != 1:
        return False, "이메일 형식이 올바르지 않습니다."
    local, _, domain = email.partition("@")
    if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
        return False, "이메일 형식이 올바르지 않습니다."
    if len(email) > 254:
        return False, "이메일이 너무 깁니다."
    return True, email


def validate_password(pw):
    """길이 8자 이상 + 영문/숫자 혼합. 과한 규칙은 오히려 재사용을 부르므로 최소한만 건다."""
    if not pw or len(pw) < 8:
        return False, "비밀번호는 8자 이상이어야 합니다."
    if not any(ch.isalpha() for ch in pw):
        return False, "비밀번호에 영문자를 포함해야 합니다."
    if not any(ch.isdigit() for ch in pw):
        return False, "비밀번호에 숫자를 포함해야 합니다."
    return True, ""


# -----------------------------------------------------------------------------
# 계정
# -----------------------------------------------------------------------------

def email_exists(email, db_path=None):
    with _conn(db_path) as c:
        row = c.execute("SELECT 1 FROM users WHERE email = ?", (email.lower(),)).fetchone()
    return row is not None


def create_user(email, password, shop_name, db_path=None, verified=False):
    """계정 생성. 이미 있는 이메일이면 (False, 사유)를 돌려준다."""
    ok, email_or_msg = validate_email(email)
    if not ok:
        return False, email_or_msg
    email = email_or_msg

    ok, msg = validate_password(password)
    if not ok:
        return False, msg

    shop_name = (shop_name or "").strip()
    if not shop_name:
        return False, "매장 이름을 입력해주세요."

    ok, msg = check_supported_domain(email)
    if not ok:
        return False, msg

    if email_exists(email, db_path):
        return False, "이미 가입된 이메일입니다."

    salt = _new_salt()
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO users (email, shop_name, password_hash, salt, is_verified, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (email, shop_name, _hash(password, salt), salt, int(verified), _iso(_now())),
        )
    return True, "가입이 완료되었습니다."


def verify_login(email, password, db_path=None):
    """로그인 검증. 성공하면 (True, 사용자 dict), 실패하면 (False, 사유).

    이메일이 없을 때와 비밀번호가 틀렸을 때 같은 메시지를 준다.
    메시지를 구분하면 어떤 이메일이 가입돼 있는지 알려주는 꼴이 되기 때문이다.
    """
    ok, email_or_msg = validate_email(email)
    if not ok:
        return False, "이메일 또는 비밀번호가 올바르지 않습니다."
    email = email_or_msg

    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if row is None:
        return False, "이메일 또는 비밀번호가 올바르지 않습니다."
    if not hmac.compare_digest(_hash(password, row["salt"]), row["password_hash"]):
        return False, "이메일 또는 비밀번호가 올바르지 않습니다."
    if not row["is_verified"]:
        return False, "이메일 인증이 완료되지 않았습니다. 인증 후 로그인해주세요."

    with _conn(db_path) as c:
        c.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (_iso(_now()), row["id"]))

    return True, {
        "id": row["id"], "email": row["email"], "shop_name": row["shop_name"],
        "created_at": row["created_at"],
    }


def mark_verified(email, db_path=None):
    with _conn(db_path) as c:
        c.execute("UPDATE users SET is_verified = 1 WHERE email = ?", (email.lower(),))


def set_password(email, new_password, db_path=None):
    ok, msg = validate_password(new_password)
    if not ok:
        return False, msg
    salt = _new_salt()
    with _conn(db_path) as c:
        cur = c.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE email = ?",
            (_hash(new_password, salt), salt, email.lower()),
        )
        if cur.rowcount == 0:
            return False, "존재하지 않는 계정입니다."
    return True, "비밀번호가 변경되었습니다."


# -----------------------------------------------------------------------------
# 이메일 인증코드
# -----------------------------------------------------------------------------

def issue_code(email, purpose="signup", db_path=None):
    """6자리 인증코드를 발급한다. 반환: (성공여부, 코드 또는 사유)

    코드 자체는 호출자(메일 발송기)에게만 돌려주고, DB에는 해시만 남긴다.
    같은 이메일로 연달아 요청하면 쿨다운으로 막는다 (메일 폭탄 방지).
    """
    email = email.lower().strip()
    now = _now()

    with _conn(db_path) as c:
        row = c.execute("SELECT sent_at FROM verifications WHERE email = ?", (email,)).fetchone()
        if row is not None:
            elapsed = (now - datetime.strptime(row["sent_at"], "%Y-%m-%d %H:%M:%S")).total_seconds()
            if elapsed < CODE_RESEND_COOLDOWN_SEC:
                wait = int(CODE_RESEND_COOLDOWN_SEC - elapsed)
                return False, f"{wait}초 후에 다시 요청할 수 있습니다."

    code = f"{secrets.randbelow(1_000_000):06d}"
    salt = _new_salt()
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO verifications (email, code_hash, salt, purpose, expires_at, attempts, sent_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?) "
            "ON CONFLICT(email) DO UPDATE SET "
            "  code_hash=excluded.code_hash, salt=excluded.salt, purpose=excluded.purpose, "
            "  expires_at=excluded.expires_at, attempts=0, sent_at=excluded.sent_at",
            (email, _hash(code, salt), salt, purpose,
             _iso(now + timedelta(minutes=CODE_TTL_MINUTES)), _iso(now)),
        )
    return True, code


def check_code(email, code, db_path=None):
    """인증코드 검증. 성공하면 해당 레코드를 삭제해 재사용을 막는다."""
    email = email.lower().strip()
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM verifications WHERE email = ?", (email,)).fetchone()

    if row is None:
        return False, "먼저 인증코드를 요청해주세요."

    if _now() > datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S"):
        with _conn(db_path) as c:
            c.execute("DELETE FROM verifications WHERE email = ?", (email,))
        return False, "인증코드가 만료되었습니다. 다시 요청해주세요."

    if row["attempts"] >= MAX_CODE_ATTEMPTS:
        with _conn(db_path) as c:
            c.execute("DELETE FROM verifications WHERE email = ?", (email,))
        return False, "시도 횟수를 초과했습니다. 인증코드를 다시 요청해주세요."

    if not hmac.compare_digest(_hash((code or "").strip(), row["salt"]), row["code_hash"]):
        with _conn(db_path) as c:
            c.execute("UPDATE verifications SET attempts = attempts + 1 WHERE email = ?", (email,))
        left = MAX_CODE_ATTEMPTS - (row["attempts"] + 1)
        return False, f"인증코드가 일치하지 않습니다. (남은 시도 {max(left, 0)}회)"

    with _conn(db_path) as c:
        c.execute("DELETE FROM verifications WHERE email = ?", (email,))
    return True, "이메일 인증이 완료되었습니다."


# -----------------------------------------------------------------------------
# 사용자별 데이터 격리
# -----------------------------------------------------------------------------

def user_data_dir(user_id):
    """소상공인마다 분리된 업로드 폴더. 남의 데이터가 섞이지 않게 하는 최소 장치."""
    d = Path(__file__).resolve().parent.parent / "data" / "users" / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_dataset(user_id, filename, stored_path, n_rows, db_path=None):
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO user_datasets (user_id, filename, stored_path, n_rows, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, filename, str(stored_path), int(n_rows), _iso(_now())),
        )


def list_datasets(user_id, db_path=None):
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT filename, stored_path, n_rows, uploaded_at FROM user_datasets "
            "WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# 계정 관리 — 매장명 / 비밀번호 / 이메일 변경, 회원 탈퇴
# -----------------------------------------------------------------------------

def get_user(user_id, db_path=None):
    """로그인 세션에 담긴 정보가 오래됐을 수 있으므로, 변경 후에는 DB에서 다시 읽는다."""
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT id, email, shop_name, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def verify_password_by_id(user_id, password, db_path=None):
    """민감한 변경(비밀번호 변경·이메일 변경·탈퇴) 전에 본인이 맞는지 다시 확인한다.

    로그인된 화면을 잠깐 자리 비운 사이 남이 조작하는 것을 막기 위한 재확인 단계다.
    """
    with _conn(db_path) as c:
        row = c.execute("SELECT password_hash, salt FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return False
    return hmac.compare_digest(_hash(password, row["salt"]), row["password_hash"])


def update_shop_name(user_id, shop_name, db_path=None):
    shop_name = (shop_name or "").strip()
    if not shop_name:
        return False, "매장 이름을 입력해주세요."
    if len(shop_name) > 60:
        return False, "매장 이름은 60자 이내로 입력해주세요."
    with _conn(db_path) as c:
        c.execute("UPDATE users SET shop_name = ? WHERE id = ?", (shop_name, user_id))
    return True, "매장 이름이 변경되었습니다."


def change_password(user_id, current_password, new_password, db_path=None):
    """현재 비밀번호를 확인한 뒤 새 비밀번호로 바꾼다. salt도 새로 만든다."""
    if not verify_password_by_id(user_id, current_password, db_path):
        return False, "현재 비밀번호가 올바르지 않습니다."
    ok, msg = validate_password(new_password)
    if not ok:
        return False, msg
    if current_password == new_password:
        return False, "현재 비밀번호와 다른 비밀번호를 입력해주세요."

    salt = _new_salt()
    with _conn(db_path) as c:
        c.execute("UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                  (_hash(new_password, salt), salt, user_id))
    return True, "비밀번호가 변경되었습니다."


def change_email(user_id, new_email, db_path=None):
    """이메일(로그인 아이디)을 바꾼다.

    호출 전에 반드시 새 주소로 보낸 인증코드를 check_code()로 통과시켜야 한다.
    확인 없이 바꾸면 오타 하나로 계정에 영영 못 들어가게 된다.
    """
    ok, email_or_msg = validate_email(new_email)
    if not ok:
        return False, email_or_msg
    new_email = email_or_msg

    with _conn(db_path) as c:
        cur = c.execute("SELECT id FROM users WHERE email = ?", (new_email,)).fetchone()
        if cur is not None and cur["id"] != user_id:
            return False, "이미 다른 계정이 사용 중인 이메일입니다."
        c.execute("UPDATE users SET email = ? WHERE id = ?", (new_email, user_id))
    return True, "이메일이 변경되었습니다."


def delete_user(user_id, password, db_path=None):
    """회원 탈퇴 — 계정과 업로드 파일을 모두 지운다.

    되돌릴 수 없으므로 비밀번호를 다시 확인한다.
    업로드 파일까지 지우는 이유는, 계정만 지우고 파일을 남기면 주인 없는
    매출 데이터가 디스크에 계속 남기 때문이다.
    """
    if not verify_password_by_id(user_id, password, db_path):
        return False, "비밀번호가 올바르지 않습니다."

    user = get_user(user_id, db_path)
    if user is None:
        return False, "이미 삭제된 계정입니다."

    # 1) 업로드 파일 폴더 삭제
    import shutil
    folder = Path(__file__).resolve().parent.parent / "data" / "users" / str(user_id)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)

    # 2) DB 레코드 삭제
    with _conn(db_path) as c:
        c.execute("DELETE FROM user_datasets WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM verifications WHERE email = ?", (user["email"],))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))

    return True, "탈퇴가 완료되었습니다. 그동안 이용해주셔서 감사합니다."
