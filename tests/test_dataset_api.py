from fastapi.testclient import TestClient

import backend.api.main as main_module
from backend.api.main import app
from backend.src.score import FraudScorer
from backend.src.review_store import SupabaseStoreError


def _transactions():
    return [
        {
            "transaction_id": f"uploaded-{index}",
            "timestamp": f"2026-08-01T10:{index:02d}:00Z",
            "user_id": "uploaded-user",
            "device_id": "uploaded-device",
            "card_id": "uploaded-card",
            "amount": 100.0 + index,
            "billing_country": "IN",
            "ip_country": "IN",
            "merchant_category": "grocery",
        }
        for index in range(3)
    ]


def test_independent_batch_uploads_can_score_the_same_timestamps_twice(monkeypatch):
    monkeypatch.setattr(main_module, "scorer", FraudScorer())
    client = TestClient(app)

    first = client.post("/score/batch", json=_transactions())
    second = client.post("/score/batch", json=_transactions())

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(first.json()) == len(second.json()) == 3


def test_batch_results_include_reviewer_parameters_from_backend(monkeypatch):
    monkeypatch.setattr(main_module, "scorer", FraudScorer())

    response = TestClient(app).post(
        "/score/batch",
        json=[{**_transactions()[0], "actual": "Legitimate"}],
    )

    assert response.status_code == 200
    row = response.json()[0]
    assert {
        "transaction_id", "amount", "velocity", "ip_billing", "device",
        "amount_deviation", "hour", "score", "status", "actual",
    } <= row.keys()
    assert row["transaction_id"] == "uploaded-0"
    assert row["amount"] == 100.0
    assert row["hour"] == 10
    assert row["ip_billing"] == "Match"
    assert row["amount_deviation"] == 0.0
    assert row["device"] == "New"
    assert row["actual"] == "Legitimate"
    assert row["status"] in {"Flagged", "Not flagged"}


def test_dataset_score_endpoint_runs_model_and_persists_results(monkeypatch):
    saved = {}

    class FakeDatasetStore:
        def save_scored_dataset(self, filename, scored):
            saved["filename"] = filename
            saved["scored"] = scored.copy()
            return "73d0f968-bf4b-44f2-84f7-98dddb805743"

    monkeypatch.setattr(main_module, "_dataset_store", lambda: FakeDatasetStore(), raising=False)
    response = TestClient(app).post(
        "/datasets/score",
        json={"filename": "transactions.csv", "transactions": _transactions()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "73d0f968-bf4b-44f2-84f7-98dddb805743"
    assert body["filename"] == "transactions.csv"
    assert body["row_count"] == 3
    assert len(body["results"]) == 3
    assert saved["filename"] == "transactions.csv"
    assert saved["scored"]["transaction_id"].tolist() == [
        "uploaded-0", "uploaded-1", "uploaded-2"
    ]
    assert body["storage_status"] == "saved"
    assert body["storage_error"] is None
    assert abs(sum(body["signal_importance_percent"].values()) - 100.0) < 0.2
    assert body["decision_threshold"] == main_module.scorer.threshold


def test_dataset_signal_importance_tracks_uploaded_model_inputs(monkeypatch):
    class FakeDatasetStore:
        def save_scored_dataset(self, _filename, _scored):
            return "dataset-test"

    monkeypatch.setattr(main_module, "_dataset_store", lambda: FakeDatasetStore())
    monkeypatch.setattr(main_module, "scorer", FraudScorer())
    client = TestClient(app)

    matching = client.post(
        "/datasets/score",
        json={"filename": "matching.csv", "transactions": _transactions()},
    )
    mismatching = client.post(
        "/datasets/score",
        json={
            "filename": "mismatching.csv",
            "transactions": [
                {**transaction, "ip_country": "US"}
                for transaction in _transactions()
            ],
        },
    )

    assert matching.status_code == mismatching.status_code == 200
    matching_importance = matching.json()["signal_importance_percent"]
    mismatching_importance = mismatching.json()["signal_importance_percent"]
    assert matching_importance != mismatching_importance
    assert mismatching_importance["Geography mismatch"] > matching_importance["Geography mismatch"]


def test_dataset_score_returns_model_results_when_supabase_storage_is_unavailable(monkeypatch):
    class UnavailableDatasetStore:
        def save_scored_dataset(self, filename, scored):
            raise SupabaseStoreError("fraud_datasets table is not available")

    monkeypatch.setattr(main_module, "_dataset_store", lambda: UnavailableDatasetStore())

    response = TestClient(app).post(
        "/datasets/score",
        json={"filename": "transactions.csv", "transactions": _transactions()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] is None
    assert body["storage_status"] == "unavailable"
    assert "fraud_datasets" in body["storage_error"]
    assert body["row_count"] == 3
    assert len(body["results"]) == 3


def test_preview_chat_endpoint_works_without_azure_configuration(monkeypatch):
    for name in (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)

    response = TestClient(app).post(
        "/preview-chat",
        json={
            "transaction": {
                "payment_id": "pay_demo",
                "amount": 32000,
                "currency": "INR",
                "risk_score": .88,
                "velocity": 12,
                "new_device": True,
            },
            "question": "Summarize risk factors",
            "history": [],
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "generated"
    assert response.json()["provider"] == "grounded-rules"
