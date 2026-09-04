from fastapi.testclient import TestClient

import backend.api.main as main_module
from backend.api.main import app
from backend.src.case_store import CaseStore


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
        self.calls.append({"method": method, "params": params, "json": json})
        return self.responses.pop(0)


def _install_fake_case_store(monkeypatch, responses):
    http = FakeHTTPClient(responses)
    store = CaseStore("https://example.supabase.co", "server-secret", http_client=http)
    monkeypatch.setattr(main_module, "_case_store", lambda: store)
    return http


def test_set_case_status_endpoint_rejects_unknown_status(monkeypatch):
    _install_fake_case_store(monkeypatch, [])

    response = TestClient(app).post(
        "/cases/pay_demo_0006/status",
        json={"status": "not_a_real_status", "actor": "analyst@example.com"},
    )

    assert response.status_code == 422


def test_set_case_status_endpoint_upserts_and_returns_the_case(monkeypatch):
    _install_fake_case_store(monkeypatch, [
        FakeResponse([{
            "transaction_id": "pay_demo_0006", "status": "confirmed_fraud",
            "risk_score": 0.91, "updated_by": "analyst@example.com",
        }]),
    ])

    response = TestClient(app).post(
        "/cases/pay_demo_0006/status",
        json={"status": "confirmed_fraud", "actor": "analyst@example.com", "risk_score": 0.91},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "confirmed_fraud"


def test_add_case_note_endpoint_rejects_blank_note(monkeypatch):
    _install_fake_case_store(monkeypatch, [])

    response = TestClient(app).post("/cases/pay_demo_0006/notes", json={"note": "   "})

    assert response.status_code == 422


def test_add_case_note_endpoint_saves_a_note(monkeypatch):
    http = _install_fake_case_store(monkeypatch, [
        FakeResponse(None, status_code=201),
        FakeResponse([{"transaction_id": "pay_demo_0006", "note": "Called the customer.", "author": "ana"}]),
    ])

    response = TestClient(app).post(
        "/cases/pay_demo_0006/notes", json={"note": "Called the customer.", "author": "ana"},
    )

    assert response.status_code == 200
    assert response.json()["note"] == "Called the customer."
    assert len(http.calls) == 2


def test_get_case_returns_case_and_notes(monkeypatch):
    _install_fake_case_store(monkeypatch, [
        FakeResponse([{"transaction_id": "pay_demo_0006", "status": "open"}]),
        FakeResponse([{"transaction_id": "pay_demo_0006", "note": "First look."}]),
    ])

    response = TestClient(app).get("/cases/pay_demo_0006")

    body = response.json()
    assert body["case"]["status"] == "open"
    assert body["notes"][0]["note"] == "First look."


def test_get_case_returns_none_case_when_no_case_exists_yet(monkeypatch):
    _install_fake_case_store(monkeypatch, [FakeResponse([]), FakeResponse([])])

    response = TestClient(app).get("/cases/pay_demo_9999")

    assert response.json() == {"case": None, "notes": []}


def test_list_cases_endpoint_forwards_status_filter(monkeypatch):
    http = _install_fake_case_store(monkeypatch, [
        FakeResponse([{"transaction_id": "pay_demo_0006", "status": "confirmed_fraud"}]),
    ])

    response = TestClient(app).get("/cases", params={"status": "confirmed_fraud"})

    assert response.json()[0]["status"] == "confirmed_fraud"
    assert http.calls[0]["params"]["status"] == "eq.confirmed_fraud"


def test_case_endpoints_return_503_when_supabase_is_not_configured(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "")

    response = TestClient(app).get("/cases")

    assert response.status_code == 503
