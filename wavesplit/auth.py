from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from fastapi import Request

from .config import AppConfig, AuthUserConfig


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def configured_user(config: AppConfig, username: str) -> AuthUserConfig | None:
    for user in config.auth.users:
        if hmac.compare_digest(user.username, username):
            return user
    return None


def password_matches(user: AuthUserConfig, password: str) -> bool:
    return hmac.compare_digest(user.password, password)


def create_session_token(config: AppConfig, username: str) -> str:
    payload = {
        "username": username,
        "expires_at": int(time.time()) + int(config.auth.session_max_age_sec),
    }
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        config.auth.session_secret.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64encode(signature)}"


def verify_session_token(config: AppConfig, token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    payload_b64, signature_b64 = token.rsplit(".", 1)
    expected = hmac.new(
        config.auth.session_secret.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        signature = _b64decode(signature_b64)
        payload = json.loads(_b64decode(payload_b64).decode("utf-8"))
    except Exception:
        return None
    if not hmac.compare_digest(signature, expected):
        return None
    if int(payload.get("expires_at") or 0) < int(time.time()):
        return None
    username = payload.get("username")
    if not isinstance(username, str) or configured_user(config, username) is None:
        return None
    return username


def username_from_request(config: AppConfig, request: Request) -> str | None:
    return verify_session_token(config, request.cookies.get(config.auth.session_cookie))
