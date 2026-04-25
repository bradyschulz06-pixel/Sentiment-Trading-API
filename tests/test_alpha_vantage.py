import pytest
from unittest.mock import MagicMock, patch

from app.services.alpha_vantage import _check_av_response


def test_check_av_response_raises_on_information_key() -> None:
    payload = {"Information": "Thank you for using Alpha Vantage! Your API call frequency is 5 calls per minute."}
    with pytest.raises(RuntimeError, match="Alpha Vantage EARNINGS"):
        _check_av_response(payload, "EARNINGS")


def test_check_av_response_raises_on_note_key() -> None:
    payload = {"Note": "Thank you for using Alpha Vantage! Please consider upgrading your API key."}
    with pytest.raises(RuntimeError, match="Alpha Vantage EARNINGS_CALL_TRANSCRIPT"):
        _check_av_response(payload, "EARNINGS_CALL_TRANSCRIPT")


def test_check_av_response_raises_on_error_message_key() -> None:
    payload = {"Error Message": "Invalid API call. Please retry or visit documentation."}
    with pytest.raises(RuntimeError, match="Alpha Vantage EARNINGS_CALENDAR"):
        _check_av_response(payload, "EARNINGS_CALENDAR")


def test_check_av_response_passes_for_valid_payload() -> None:
    payload = {"quarterlyEarnings": [{"fiscalDateEnding": "2026-03-31", "reportedEPS": "2.50"}]}
    _check_av_response(payload, "EARNINGS")  # Should not raise


def test_check_av_response_passes_for_empty_payload() -> None:
    _check_av_response({}, "EARNINGS")  # Should not raise
