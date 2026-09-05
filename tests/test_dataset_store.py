import pandas as pd

from backend.src.dataset_store import DatasetStore


class FakeResponse:
    def __init__(self, body=None, status_code=201):
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
        self.calls.append({
            "method": method,
            "url": url,
            "params": params,
            "json": json,
            "headers": headers,
            "timeout": timeout,
        })
        return self.responses.pop(0)


def _scored_frame():
    return pd.DataFrame([
        {
            "transaction_id": "txn-1",
            "timestamp": pd.Timestamp("2026-09-05T10:00:00Z"),
            "user_id": "user-1",
            "device_id": "device-1",
            "card_id": "card-1",
            "amount": 4999.0,
            "billing_country": "IN",
            "ip_country": "RU",
            "merchant_category": "electronics",
            "score": .91,
            "flagged": True,
            "blocked": True,
            "reasons": ["Billing country IN, but IP country RU"],
        }
    ])


def test_dataset_store_persists_metadata_rows_and_completion():
    http = FakeHTTPClient([FakeResponse(), FakeResponse(), FakeResponse(status_code=204)])
    store = DatasetStore("https://example.supabase.co", "server-secret", http_client=http)

    dataset_id = store.save_scored_dataset("../incoming/transactions.csv", _scored_frame())

    assert dataset_id
    assert [call["method"] for call in http.calls] == ["POST", "POST", "PATCH"]
    assert http.calls[0]["json"]["filename"] == "transactions.csv"
    assert http.calls[0]["json"]["status"] == "processing"
    saved_row = http.calls[1]["json"][0]
    assert saved_row["dataset_id"] == dataset_id
    assert saved_row["transaction_id"] == "txn-1"
    assert saved_row["score"] == .91
    assert saved_row["reasons"] == ["Billing country IN, but IP country RU"]
    assert http.calls[2]["params"] == {"id": f"eq.{dataset_id}"}
    assert http.calls[2]["json"]["status"] == "completed"


def test_dataset_store_persists_reviewer_parameters():
    http = FakeHTTPClient([FakeResponse(), FakeResponse(), FakeResponse(status_code=204)])
    store = DatasetStore("https://example.supabase.co", "server-secret", http_client=http)
    scored = _scored_frame().assign(
        velocity=[3.0],
        ip_billing=["Mismatch"],
        device=["New"],
        amount_deviation=[1.75],
        hour=[10],
        status=["Flagged"],
        actual=["Fraud"],
    )

    store.save_scored_dataset("transactions.csv", scored)

    saved_row = http.calls[1]["json"][0]
    assert {
        key: saved_row[key]
        for key in (
            "velocity", "ip_billing", "device", "amount_deviation", "hour", "status", "actual"
        )
    } == {
        "velocity": 3.0,
        "ip_billing": "Mismatch",
        "device": "New",
        "amount_deviation": 1.75,
        "hour": 10,
        "status": "Flagged",
        "actual": "Fraud",
    }
