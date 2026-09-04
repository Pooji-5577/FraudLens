import pytest

from backend.src.case_store import CaseStore
from backend.src.review_store import SupabaseStoreError


class FakeResponse:
    def __init__(self, body=None, status_code=200):
        self.status_code = status_code
        self._body = body
        self.content = b"" if body is None else b"response"

    def json(self):
        return self._body


class FakeHTTPClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, *, params, json, headers, timeout):
        self.calls.append({"method": method, "url": url, "params": params, "json": json, "headers": headers})
        return self.responses.pop(0)


def case_row(transaction_id="pay_demo_0006", status="open"):
    return {
        "transaction_id": transaction_id,
        "status": status,
        "risk_score": 0.912,
        "created_at": "2026-09-04T00:00:00+00:00",
        "updated_at": "2026-09-04T00:00:00+00:00",
        "updated_by": "analyst@example.com",
    }


def test_set_status_upserts_with_on_conflict_and_merge_duplicates():
    http = FakeHTTPClient([FakeResponse([case_row(status="under_investigation")])])
    store = CaseStore("https://example.supabase.co", "server-secret", http_client=http)

    result = store.set_status(
        "pay_demo_0006", "under_investigation", actor="analyst@example.com", risk_score=0.912
    )

    assert result["status"] == "under_investigation"
    call = http.calls[0]
    assert call["method"] == "POST"
    assert call["params"] == {"on_conflict": "transaction_id"}
    assert call["json"]["status"] == "under_investigation"
    assert call["json"]["risk_score"] == 0.912


def test_set_status_rejects_unknown_status():
    store = CaseStore("https://example.supabase.co", "server-secret", http_client=FakeHTTPClient([]))

    with pytest.raises(ValueError, match="status must be one of"):
        store.set_status("pay_demo_0006", "not_a_real_status")


def test_add_note_ensures_case_exists_without_resetting_its_status():
    http = FakeHTTPClient([
        FakeResponse(None, status_code=201),  # ensure-exists upsert (ignore-duplicates)
        FakeResponse([{"transaction_id": "pay_demo_0006", "note": "Called the customer.", "author": "ana"}]),
    ])
    store = CaseStore("https://example.supabase.co", "server-secret", http_client=http)

    result = store.add_note("pay_demo_0006", "Called the customer.", author="ana")

    assert result["note"] == "Called the customer."
    ensure_call, note_call = http.calls
    assert ensure_call["json"]["status"] == "open"
    assert "resolution=ignore-duplicates" in ensure_call["headers"]["Prefer"]
    assert note_call["json"]["note"] == "Called the customer."


def test_add_note_rejects_blank_note():
    store = CaseStore("https://example.supabase.co", "server-secret", http_client=FakeHTTPClient([]))

    with pytest.raises(ValueError, match="must not be empty"):
        store.add_note("pay_demo_0006", "   ")


def test_list_cases_filters_by_status_when_given():
    http = FakeHTTPClient([FakeResponse([case_row(status="confirmed_fraud")])])
    store = CaseStore("https://example.supabase.co", "server-secret", http_client=http)

    result = store.list_cases(status="confirmed_fraud")

    assert result[0]["status"] == "confirmed_fraud"
    assert http.calls[0]["params"]["status"] == "eq.confirmed_fraud"


def test_get_case_returns_none_when_missing():
    http = FakeHTTPClient([FakeResponse([])])
    store = CaseStore("https://example.supabase.co", "server-secret", http_client=http)

    assert store.get_case("pay_demo_9999") is None


def test_from_environment_requires_url_and_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "")

    with pytest.raises(SupabaseStoreError, match="SUPABASE_URL"):
        CaseStore.from_environment()

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    with pytest.raises(SupabaseStoreError, match="SUPABASE_SECRET_KEY"):
        CaseStore.from_environment()
