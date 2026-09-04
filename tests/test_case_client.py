import pytest
import requests

from frontend.processing import (
    ScoringAPIError,
    add_fraud_case_note,
    get_fraud_case,
    list_fraud_cases,
    set_fraud_case_status,
)


class _Response:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(response=self)
            raise error

    def json(self):
        return self.body


class _RecordingClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"method": "GET", "url": url, "params": params})
        return self.response

    def post(self, url, json=None, timeout=None):
        self.calls.append({"method": "POST", "url": url, "json": json})
        return self.response


def test_list_fraud_cases_forwards_status_filter():
    client = _RecordingClient(_Response([{"transaction_id": "pay_demo_0006", "status": "open"}]))

    result = list_fraud_cases("http://scoring.test", status="open", http_client=client)

    assert result[0]["status"] == "open"
    assert client.calls[0]["params"] == {"status": "open"}


def test_list_fraud_cases_raises_on_invalid_payload():
    client = _RecordingClient(_Response({"not": "a list"}))

    with pytest.raises(ScoringAPIError, match="invalid case list"):
        list_fraud_cases("http://scoring.test", http_client=client)


def test_get_fraud_case_returns_case_and_notes():
    client = _RecordingClient(_Response({"case": {"status": "open"}, "notes": []}))

    result = get_fraud_case("pay_demo_0006", "http://scoring.test", http_client=client)

    assert result["case"]["status"] == "open"
    assert client.calls[0]["url"] == "http://scoring.test/cases/pay_demo_0006"


def test_set_fraud_case_status_sends_actor_and_risk_score():
    client = _RecordingClient(_Response({"status": "confirmed_fraud"}))

    result = set_fraud_case_status(
        "pay_demo_0006", "confirmed_fraud", "http://scoring.test",
        actor="analyst@example.com", risk_score=0.91, http_client=client,
    )

    assert result["status"] == "confirmed_fraud"
    assert client.calls[0]["json"] == {
        "status": "confirmed_fraud", "actor": "analyst@example.com", "risk_score": 0.91,
    }


def test_add_fraud_case_note_sends_note_and_author():
    client = _RecordingClient(_Response({"note": "Called the customer."}))

    result = add_fraud_case_note(
        "pay_demo_0006", "Called the customer.", "http://scoring.test",
        author="ana", http_client=client,
    )

    assert result["note"] == "Called the customer."
    assert client.calls[0]["json"] == {"note": "Called the customer.", "author": "ana"}


def test_case_functions_surface_server_error_detail():
    client = _RecordingClient(_Response({"detail": "Supabase is unreachable."}, status_code=503))

    with pytest.raises(ScoringAPIError, match="Supabase is unreachable."):
        list_fraud_cases("http://scoring.test", http_client=client)
