from app.auth import create_password_hash, verify_password


def test_password_hash_round_trip() -> None:
    hashed = create_password_hash("northstar-secret")
    assert verify_password("northstar-secret", hashed)
    assert not verify_password("wrong-password", hashed)


def test_plaintext_fallback_still_works_for_local_dev() -> None:
    assert verify_password("change-me-now", "change-me-now")
    assert not verify_password("different", "change-me-now")
