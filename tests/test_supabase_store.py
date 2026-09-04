import pytest

from backend.src.review_store import (
    SupabaseRESTClient,
    SupabaseReviewStore,
    SupabaseStoreError,
    review_store_from_environment,
)
from backend.src.razorpay_enforcement import ReviewStore


class FakeResponse:
    def __init__(self, body=None, status_code=200):
        self.status_code = status_code
        self._body = body
        self.content = b"" if body is None else b"response"

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeHTTPClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, *, params, json, headers, timeout):
        self.calls.append({
            "method": method,
            "url": url,
            "params": params,
            "json": json,
            "headers": headers,
            "timeout": timeout,
        })
        return self.responses.pop(0)


def review_row(payment_id="pay_1", review_status="pending_review"):
    return {
        "payment_id": payment_id,
        "order_id": "order_1",
        "amount": 12500,
        "currency": "INR",
        "payment_status": "authorized",
        "review_status": review_status,
        "fulfillment_status": "on_hold",
        "risk_score": None,
        "evidence_json": {"signals": ["velocity"]},
        "created_at": "2026-09-04T00:00:00+00:00",
        "updated_at": "2026-09-04T00:00:00+00:00",
        "decided_at": None,
        "decided_by": None,
        "decision": None,
    }


def authorized_payload(payment_id="pay_1"):
    return {
        "event": "payment.authorized",
        "payload": {"payment": {"entity": {
            "id": payment_id,
            "order_id": "order_1",
            "amount": 12500,
            "currency": "INR",
            "status": "authorized",
            "captured": False,
        }}},
    }


def test_rest_client_sends_server_key_as_apikey_only():
    http = FakeHTTPClient([FakeResponse([])])
    client = SupabaseRESTClient("https://example.supabase.co", "server-secret", http_client=http)

    assert client.request("GET", "payment_reviews", params={"limit": "1"}) == []
    assert http.calls[0]["headers"] == {
        "apikey": "server-secret",
        "Content-Type": "application/json",
    }


def test_supabase_store_processes_and_deduplicates_webhooks():
    http = FakeHTTPClient([
        FakeResponse([{"event_id": "event_1"}]),
        FakeResponse([]),
        FakeResponse([review_row()]),
        FakeResponse([]),
    ])
    store = SupabaseReviewStore("https://example.supabase.co", "server-secret", http_client=http)

    first = store.process_event("event_1", authorized_payload())
    duplicate = store.process_event("event_1", authorized_payload())

    assert first == {"status": "pending_review", "duplicate": False}
    assert duplicate == {"status": "duplicate", "duplicate": True}
    assert [call["method"] for call in http.calls] == ["POST", "GET", "POST", "POST"]
    assert http.calls[2]["json"]["payment_id"] == "pay_1"


def test_supabase_store_maps_json_evidence_and_writes_audit():
    http = FakeHTTPClient([
        FakeResponse([review_row()]),
        FakeResponse([review_row()]),
        FakeResponse(None, status_code=201),
    ])
    store = SupabaseReviewStore("https://example.supabase.co", "server-secret", http_client=http)

    review = store.get_review("pay_1")
    store.append_audit(
        "pay_1",
        action="capture",
        actor="reviewer@example.com",
        outcome="failed",
        detail="test",
    )

    assert review["evidence"] == {"signals": ["velocity"]}
    assert http.calls[2]["json"]["evidence_json"] == {"signals": ["velocity"]}
    assert http.calls[2]["json"]["actor"] == "reviewer@example.com"


def test_supabase_store_requires_server_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    with pytest.raises(SupabaseStoreError, match="server-side secret key"):
        SupabaseReviewStore.from_environment()


def test_store_factory_is_sqlite_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("FRAUDLENS_STORAGE", raising=False)

    store = review_store_from_environment(tmp_path / "reviews.sqlite3")

    assert isinstance(store, ReviewStore)


def test_store_factory_can_select_supabase(monkeypatch):
    monkeypatch.setenv("FRAUDLENS_STORAGE", "supabase")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "server-secret")

    store = review_store_from_environment("ignored.sqlite3")

    assert isinstance(store, SupabaseReviewStore)
