import pandas as pd
from pathlib import Path

from backend.src.features import engineer_features
import backend.src.report as report_module
from backend.src.report import (
    CHAT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    answer_preview_transaction_question,
    answer_transaction_question,
    demo_evidence_from_signals,
    evidence_from_features,
    generate_demo_report,
    generate_report,
)
from backend.src.score import FraudScorer


AZURE_ENV = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION",
)


def test_azure_configuration_is_loaded_from_the_repository_root():
    expected = Path(__file__).resolve().parents[1] / ".env"
    assert report_module.PROJECT_ENV_PATH == expected


def _transactions():
    base = pd.Timestamp("2026-08-01T10:00:00Z")
    amounts = [100.0, 140.0, 500.0]
    return pd.DataFrame([
        {
            "transaction_id": f"report-{i}",
            "timestamp": (base + pd.Timedelta(minutes=10 * i)).isoformat(),
            "user_id": "report-user",
            "device_id": "known-device" if i < 2 else "new-device",
            "card_id": "report-card",
            "amount": amount,
            "billing_country": "IN",
            "ip_country": "RU" if i == 2 else "IN",
            "merchant_category": "electronics",
        }
        for i, amount in enumerate(amounts)
    ])


def test_scoring_shape_is_unchanged_when_report_feature_is_disabled(monkeypatch):
    for name in AZURE_ENV:
        monkeypatch.delenv(name, raising=False)
    result = FraudScorer().score_one(_transactions().iloc[0].to_dict(), include_report=True)
    assert set(result) == {"score", "flagged", "blocked", "reasons"}
    assert result["blocked"] is result["flagged"]


def test_report_evidence_values_equal_computed_features_exactly():
    row = engineer_features(_transactions()).iloc[-1]
    evidence = {item["signal"]: item for item in evidence_from_features(row.to_dict())}
    assert evidence["card_velocity"]["values"] == {
        "count_1h": int(row["card_txn_count_1h"]),
        "count_24h": int(row["card_txn_count_24h"]),
    }
    assert evidence["geography"]["values"] == {
        "billing_country": row["billing_country"],
        "ip_country": row["ip_country"],
        "mismatch": bool(row["geo_mismatch"]),
    }
    assert evidence["amount_deviation"]["values"] == {
        "amount": float(row["amount"]),
        "prior_mean": float(row["user_amount_mean"]),
        "prior_std": float(row["user_amount_std"]),
        "zscore": float(row["user_amount_zscore"]),
    }
    assert evidence["device_history"]["values"]["is_new_device"] is bool(row["is_new_device"])


def test_demo_report_is_grounded_in_visible_synthetic_signals(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "mock-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "review-summary")
    transaction = {
        "payment_id": "pay_demo_1",
        "velocity": 17,
        "ip_billing_mismatch": True,
        "new_device": True,
        "amount_deviation": 72,
        "risk_score": .91,
    }
    captured = {}

    def writer(_config, evidence, score, threshold, reasons):
        captured.update(evidence=evidence, score=score, threshold=threshold, reasons=reasons)
        return "Recent activity was elevated and the payment used a new device."

    report = generate_demo_report(transaction, threshold=.65, summary_writer=writer)

    assert report["status"] == "generated"
    assert report["summary"].startswith("Recent activity")
    assert report["recommended_action"] == "demo-block"
    assert report["evidence"] == demo_evidence_from_signals(transaction)
    assert captured["score"] == .91
    assert captured["threshold"] == .65


def test_azure_failure_returns_verified_evidence_without_secret(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "do-not-expose-this-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "review-summary")
    row = engineer_features(_transactions()).iloc[-1].to_dict()

    def fail(*_args):
        raise RuntimeError("provider failed and mentioned do-not-expose-this-secret")

    report = generate_report(row, .82, True, .26, ["Billing country IN, but IP country RU"], fail)
    assert report["status"] == "failed"
    assert report["summary"] is None
    assert report["evidence"] == evidence_from_features(row)
    assert "do-not-expose-this-secret" not in report["error"]
    assert report["recommended_action"] == "auto-block"


def test_azure_request_is_mocked_and_grounded_in_verified_evidence(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "mock-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "review-summary")
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "The verified signals triggered review. A human should assess the payment."}}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(report_module.requests, "post", fake_post)
    row = engineer_features(_transactions()).iloc[-1].to_dict()
    generated = generate_report(row, .82, True, .26, ["Billing country IN, but IP country RU"])

    assert generated["status"] == "generated"
    messages = captured["json"]["messages"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "never add" in messages[0]["content"].lower()
    assert "Billing country IN and IP country RU do not match." in messages[1]["content"]
    assert f"Transaction amount {row['amount']:.2f}" in messages[1]["content"]
    assert captured["headers"]["api-key"] == "mock-key"


def test_transaction_chat_is_grounded_in_selected_scored_context(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "mock-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "transaction-chat")
    row = engineer_features(_transactions()).iloc[-1].to_dict()
    captured = {}

    def fake_writer(config, context, question, history):
        captured.update({
            "config": config,
            "context": context,
            "question": question,
            "history": history,
        })
        return "It was blocked because the score exceeded the threshold and the countries differed."

    result = answer_transaction_question(
        row,
        .82,
        True,
        .26,
        ["Billing country IN, but IP country RU"],
        "Why was this blocked?",
        [{"role": "user", "content": "What is the amount?"}],
        fake_writer,
    )

    assert "only the verified transaction context" in CHAT_SYSTEM_PROMPT
    assert captured["context"]["transaction"]["transaction_id"] == row["transaction_id"]
    assert captured["context"]["model"] == {
        "score": .82,
        "blocking_threshold": .26,
        "blocked": True,
        "reasons": ["Billing country IN, but IP country RU"],
    }
    assert captured["context"]["verified_evidence"] == evidence_from_features(row)
    assert captured["question"] == "Why was this blocked?"
    assert result["status"] == "generated"


def test_preview_chat_uses_raw_fields_and_marks_model_context_unavailable(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "mock-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "transaction-chat")
    transaction = _transactions().iloc[0].to_dict()
    captured = {}

    def fake_writer(config, context, question, history):
        captured["context"] = context
        return f"The amount is {context['transaction']['amount']:.2f}."

    result = answer_preview_transaction_question(
        transaction,
        "What is the amount?",
        answer_writer=fake_writer,
    )

    assert captured["context"]["transaction"]["transaction_id"] == transaction["transaction_id"]
    assert captured["context"]["model"]["status"] == "not scored"
    assert "No fraud score" in captured["context"]["model"]["note"]
    assert result["answer"] == "The amount is 100.00."


def test_preview_chat_accepts_razorpay_payment_fields(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "mock-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "transaction-chat")
    captured = {}
    payment = {
        "payment_id": "pay_123",
        "timestamp": "2026-09-04T10:00:00+00:00",
        "amount": 499.0,
        "currency": "INR",
        "status": "captured",
        "method": "upi",
        "order_id": "order_123",
    }

    def fake_writer(config, context, question, history):
        captured["context"] = context
        return "The payment was captured using UPI."

    result = answer_preview_transaction_question(
        payment,
        "What happened?",
        answer_writer=fake_writer,
    )

    assert captured["context"]["transaction"]["payment_id"] == "pay_123"
    assert captured["context"]["transaction"]["status"] == "captured"
    assert result["transaction_id"] == "pay_123"
