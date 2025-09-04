import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import bcrypt
import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # default 7 days


def hash_password(raw: str) -> str:
    if not isinstance(raw, str) or len(raw) < 6:
        raise ValueError("password too short")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(raw.encode("utf-8"), salt).decode("utf-8")


def verify_password(raw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(raw.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: int, email: str) -> Tuple[str, datetime]:
    now = datetime.now(tz=timezone.utc)
    exp = now + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "typ": "access",
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return token, exp


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.exceptions.PyJWTError:
        return None
