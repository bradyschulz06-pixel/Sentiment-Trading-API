from __future__ import annotations

import base64
import hashlib
import hmac
import os
import warnings


ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 120_000


def create_password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"{ALGORITHM}${ITERATIONS}${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def verify_password(password: str, stored_value: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored_value.split("$", 3)
    except ValueError:
        warnings.warn(
            "Plaintext password detected. Set ADMIN_PASSWORD_HASH to a hashed value for security.",
            stacklevel=2,
        )
        return hmac.compare_digest(password, stored_value)
    if algorithm != ALGORITHM:
        return False
    salt = base64.urlsafe_b64decode(salt_b64.encode())
    expected = base64.urlsafe_b64decode(digest_b64.encode())
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return hmac.compare_digest(candidate, expected)
