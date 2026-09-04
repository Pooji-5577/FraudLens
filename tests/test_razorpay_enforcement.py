import hashlib
import hmac
import json
import pytest

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.src.razorpay_enforcement import (
    EnforcementError,
    EnforcementService,
    RazorpayGateway,
    ReviewStore,
)


def signed_headers(body: bytes, event_id: str, secret: str = "test-webhook-secret") -> dict[str, str]:
    return {
        "content-type": "application/json",
        "x-razorpay-event-id": event_id,
        "x-razorpay-signature": hmac.new(secret.encode(), body, hashlib.sha256).hexdigest(),
    }


def test_webhook_rejects_invalid_signature_before_processing(monkeypatch, tmp_path):
    monkeypatch.setenv("RAZORPAY_MODE", "test")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("RAZORPAY_ENFORCEMENT_DB", str(tmp_path / "enforcement.sqlite3"))
    if hasattr(app.state, "razorpay_enforcement"):
        del app.state.razorpay_enforcement
    body = json.dumps({"event": "payment.authorized", "payload": {}}).encode()

    response = TestClient(app).post(
        "/webhooks/razorpay",
        content=body,
        headers={
            "content-type": "application/json",
            "x-razorpay-event-id": "event_invalid",
            "x-razorpay-signature": "not-a-valid-signature",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid Razorpay webhook signature."}
    assert not (tmp_path / "enforcement.sqlite3").exists()


def test_authorized_webhook_creates_one_pending_review_and_deduplicates(monkeypatch, tmp_path):
    database = tmp_path / "enforcement.sqlite3"
    monkeypatch.setenv("RAZORPAY_MODE", "test")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("RAZORPAY_ENFORCEMENT_DB", str(database))
    if hasattr(app.state, "razorpay_enforcement"):
        del app.state.razorpay_enforcement
    body = json.dumps({
        "event": "payment.authorized",
        "payload": {"payment": {"entity": {
            "id": "pay_authorized_1",
            "order_id": "order_1",
            "amount": 12500,
            "currency": "INR",
            "status": "authorized",
            "captured": False,
        }}},
    }, separators=(",", ":")).encode()
    client = TestClient(app)

    first = client.post("/webhooks/razorpay", content=body, headers=signed_headers(body, "event_1"))
    duplicate = client.post("/webhooks/razorpay", content=body, headers=signed_headers(body, "event_1"))

    assert first.status_code == 202
    assert first.json() == {"status": "pending_review", "duplicate": False}
    assert duplicate.status_code == 202
    assert duplicate.json() == {"status": "duplicate", "duplicate": True}
    review = ReviewStore(database).get_review("pay_authorized_1")
    assert review["review_status"] == "pending_review"
    assert review["amount"] == 12500
    assert review["currency"] == "INR"


def test_authorized_webhook_marks_an_already_captured_payment_as_not_holdable(monkeypatch, tmp_path):
    database = tmp_path / "enforcement.sqlite3"
    monkeypatch.setenv("RAZORPAY_MODE", "test")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("RAZORPAY_ENFORCEMENT_DB", str(database))
    if hasattr(app.state, "razorpay_enforcement"):
        del app.state.razorpay_enforcement
    body = json.dumps({
        "event": "payment.authorized",
        "payload": {"payment": {"entity": {
            "id": "pay_captured_early",
            "order_id": "order_2",
            "amount": 5000,
            "currency": "INR",
            "status": "captured",
            "captured": True,
        }}},
    }, separators=(",", ":")).encode()

    response = TestClient(app).post(
        "/webhooks/razorpay", content=body, headers=signed_headers(body, "event_captured")
    )

    assert response.status_code == 202
    assert response.json() == {"status": "already_captured", "duplicate": False}
    review = ReviewStore(database).get_review("pay_captured_early")
    assert review["review_status"] == "already_captured"
    assert review["fulfillment_status"] == "on_hold"


def test_authorization_revoked_webhook_persists_account_revocation(monkeypatch, tmp_path):
    database = tmp_path / "enforcement.sqlite3"
    monkeypatch.setenv("RAZORPAY_MODE", "test")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("RAZORPAY_ENFORCEMENT_DB", str(database))
    if hasattr(app.state, "razorpay_enforcement"):
        del app.state.razorpay_enforcement
    body = json.dumps({
        "event": "account.app.authorization_revoked",
        "payload": {"account": {"entity": {"id": "acc_revoked"}}},
    }, separators=(",", ":")).encode()

    response = TestClient(app).post(
        "/webhooks/razorpay", content=body, headers=signed_headers(body, "event_revoked")
    )

    assert response.status_code == 202
    assert response.json() == {
        "status": "authorization_revoked", "duplicate": False, "account_id": "acc_revoked"
    }
    assert ReviewStore(database).is_authorization_revoked("acc_revoked")


def test_authorization_revoked_webhook_requires_account_identity(monkeypatch, tmp_path):
    database = tmp_path / "enforcement.sqlite3"
    monkeypatch.setenv("RAZORPAY_MODE", "test")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("RAZORPAY_ENFORCEMENT_DB", str(database))
    if hasattr(app.state, "razorpay_enforcement"):
        del app.state.razorpay_enforcement
    body = json.dumps({"event": "account.app.authorization_revoked", "payload": {}}).encode()

    response = TestClient(app).post(
        "/webhooks/razorpay", content=body, headers=signed_headers(body, "event_missing_account")
    )

    assert response.status_code == 422
    assert "missing account.id" in response.json()["detail"]
    assert not ReviewStore(database).is_authorization_revoked("acc_revoked")


def test_captured_event_before_authorized_event_cannot_reopen_pending_review(tmp_path):
    store = ReviewStore(tmp_path / "enforcement.sqlite3")
    payment = {
        "id": "pay_out_of_order",
        "order_id": "order_3",
        "amount": 9900,
        "currency": "INR",
        "status": "captured",
        "captured": True,
    }

    captured = store.process_event(
        "event_captured_first",
        {"event": "payment.captured", "payload": {"payment": {"entity": payment}}},
    )
    authorized_late = store.process_event(
        "event_authorized_late",
        {"event": "payment.authorized", "payload": {"payment": {"entity": payment}}},
    )

    assert captured == {"status": "already_captured", "duplicate": False}
    assert authorized_late == {"status": "already_captured", "duplicate": False}
    assert store.get_review("pay_out_of_order")["review_status"] == "already_captured"


class FakeRazorpayGateway:
    def __init__(self, payment: dict, order: dict | None = None):
        self.payment = payment
        self.order = order or {
            "id": payment.get("order_id"),
            "amount": payment["amount"],
            "currency": payment["currency"],
        }
        self.capture_calls = []
        self.refund_calls = []

    def fetch_payment(self, payment_id: str) -> dict:
        assert payment_id == self.payment["id"]
        return dict(self.payment)

    def fetch_order(self, order_id: str) -> dict:
        assert order_id == self.order["id"]
        return dict(self.order)

    def capture_payment(self, payment_id: str, *, amount: int, currency: str) -> dict:
        self.capture_calls.append((payment_id, amount, currency))
        self.payment.update({"status": "captured", "captured": True})
        return dict(self.payment)

    def refund_payment(self, payment_id: str, *, idempotency_key: str) -> dict:
        self.refund_calls.append((payment_id, idempotency_key))
        self.payment.update({"status": "refunded", "amount_refunded": self.payment["amount"]})
        return {"id": "rfnd_test", "payment_id": payment_id, "status": "processed"}


def add_review(store: ReviewStore, payment: dict, event_id: str = "event_review") -> None:
    store.process_event(
        event_id,
        {"event": "payment.authorized", "payload": {"payment": {"entity": payment}}},
    )


def test_human_approval_captures_exact_order_amount_and_writes_audit_log(tmp_path):
    store = ReviewStore(tmp_path / "enforcement.sqlite3")
    payment = {
        "id": "pay_capture",
        "order_id": "order_capture",
        "amount": 12500,
        "currency": "INR",
        "status": "authorized",
        "captured": False,
    }
    add_review(store, payment)
    gateway = FakeRazorpayGateway(payment)

    result = EnforcementService(store, gateway, mode="test").approve_and_capture(
        "pay_capture", actor="reviewer@example.com"
    )

    assert result["review_status"] == "approved_captured"
    assert gateway.capture_calls == [("pay_capture", 12500, "INR")]
    audit = store.list_audit_entries("pay_capture")
    assert len(audit) == 1
    assert audit[0]["action"] == "capture"
    assert audit[0]["actor"] == "reviewer@example.com"
    assert audit[0]["outcome"] == "succeeded"


def test_capture_retry_checks_current_state_and_does_not_capture_twice(tmp_path):
    store = ReviewStore(tmp_path / "enforcement.sqlite3")
    payment = {
        "id": "pay_capture_retry",
        "order_id": "order_capture_retry",
        "amount": 2500,
        "currency": "INR",
        "status": "authorized",
        "captured": False,
    }
    add_review(store, payment)
    gateway = FakeRazorpayGateway(payment)
    service = EnforcementService(store, gateway, mode="test")

    service.approve_and_capture("pay_capture_retry", actor="reviewer@example.com")
    retry = service.approve_and_capture("pay_capture_retry", actor="reviewer@example.com")

    assert retry["review_status"] == "approved_captured"
    assert gateway.capture_calls == [("pay_capture_retry", 2500, "INR")]
    assert [entry["outcome"] for entry in store.list_audit_entries("pay_capture_retry")] == [
        "succeeded",
        "already_captured",
    ]


def test_human_fraud_confirmation_never_calls_razorpay_and_is_idempotent(tmp_path):
    store = ReviewStore(tmp_path / "enforcement.sqlite3")
    payment = {
        "id": "pay_fraud",
        "order_id": "order_fraud",
        "amount": 7400,
        "currency": "INR",
        "status": "authorized",
        "captured": False,
    }
    add_review(store, payment)
    gateway = FakeRazorpayGateway(payment)
    service = EnforcementService(store, gateway, mode="test")

    first = service.confirm_fraud("pay_fraud", actor="analyst@example.com")
    retry = service.confirm_fraud("pay_fraud", actor="analyst@example.com")

    assert first["review_status"] == "confirmed_fraud"
    assert first["fulfillment_status"] == "stopped"
    assert retry["review_status"] == "confirmed_fraud"
    assert gateway.capture_calls == []
    assert gateway.refund_calls == []
    assert [entry["outcome"] for entry in store.list_audit_entries("pay_fraud")] == [
        "succeeded",
        "already_confirmed",
    ]


def test_full_refund_uses_stable_idempotency_key_and_is_not_repeated(tmp_path):
    store = ReviewStore(tmp_path / "enforcement.sqlite3")
    payment = {
        "id": "pay_refund",
        "order_id": "order_refund",
        "amount": 19900,
        "amount_refunded": 0,
        "currency": "INR",
        "status": "captured",
        "captured": True,
    }
    add_review(store, payment)
    gateway = FakeRazorpayGateway(payment)
    service = EnforcementService(store, gateway, mode="test")

    first = service.refund_and_stop("pay_refund", actor="analyst@example.com")
    retry = service.refund_and_stop("pay_refund", actor="analyst@example.com")

    assert first["review_status"] == "refunded"
    assert first["fulfillment_status"] == "stopped"
    assert retry["review_status"] == "refunded"
    assert len(gateway.refund_calls) == 1
    assert gateway.refund_calls[0][0] == "pay_refund"
    assert gateway.refund_calls[0][1].startswith("fraudlens-refund-")
    assert [entry["outcome"] for entry in store.list_audit_entries("pay_refund")] == [
        "succeeded",
        "already_refunded",
    ]


def test_razorpay_gateway_uses_bearer_auth_and_refund_idempotency_header():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self):
            self.calls = []

        def get(self, url, *, headers, timeout):
            self.calls.append(("GET", url, headers, None, timeout))
            return Response({"id": url.rsplit("/", 1)[-1], "status": "authorized"})

        def post(self, url, *, headers, json, timeout):
            self.calls.append(("POST", url, headers, json, timeout))
            return Response({"id": "result", "status": "processed"})

    client = Client()
    gateway = RazorpayGateway("server-side-token", http_client=client)

    gateway.fetch_payment("pay_123")
    gateway.fetch_order("order_123")
    gateway.capture_payment("pay_123", amount=12500, currency="INR")
    gateway.refund_payment("pay_123", idempotency_key="fraudlens-refund-stable-key")

    bearer = {"Authorization": "Bearer server-side-token"}
    assert client.calls == [
        ("GET", "https://api.razorpay.com/v1/payments/pay_123", bearer, None, 15.0),
        ("GET", "https://api.razorpay.com/v1/orders/order_123", bearer, None, 15.0),
        (
            "POST",
            "https://api.razorpay.com/v1/payments/pay_123/capture",
            bearer,
            {"amount": 12500, "currency": "INR"},
            15.0,
        ),
        (
            "POST",
            "https://api.razorpay.com/v1/payments/pay_123/refund",
            {**bearer, "X-Refund-Idempotency": "fraudlens-refund-stable-key"},
            {},
            15.0,
        ),
    ]


def test_rejected_capture_attempt_is_recorded_in_the_durable_audit_log(tmp_path):
    store = ReviewStore(tmp_path / "enforcement.sqlite3")
    payment = {
        "id": "pay_mismatch",
        "order_id": "order_mismatch",
        "amount": 12500,
        "currency": "INR",
        "status": "authorized",
        "captured": False,
    }
    add_review(store, payment)
    gateway = FakeRazorpayGateway(
        payment,
        order={"id": "order_mismatch", "amount": 9999, "currency": "INR"},
    )

    with pytest.raises(EnforcementError, match="does not match"):
        EnforcementService(store, gateway, mode="test").approve_and_capture(
            "pay_mismatch", actor="reviewer@example.com"
        )

    audit = store.list_audit_entries("pay_mismatch")
    assert [(entry["action"], entry["outcome"]) for entry in audit] == [("capture", "failed")]


def test_confirm_fraud_rechecks_state_and_switches_to_refund_path_if_capture_won_race(tmp_path):
    store = ReviewStore(tmp_path / "enforcement.sqlite3")
    authorized = {
        "id": "pay_capture_race",
        "order_id": "order_capture_race",
        "amount": 12500,
        "currency": "INR",
        "status": "authorized",
        "captured": False,
    }
    add_review(store, authorized)
    captured = {**authorized, "status": "captured", "captured": True}

    with pytest.raises(EnforcementError, match="refund action"):
        EnforcementService(
            store, FakeRazorpayGateway(captured), mode="test"
        ).confirm_fraud("pay_capture_race", actor="reviewer@example.com")

    assert store.get_review("pay_capture_race")["review_status"] == "already_captured"
