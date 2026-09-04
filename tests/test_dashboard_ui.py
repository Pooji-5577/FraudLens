import pytest
import requests
from streamlit.testing.v1 import AppTest

import frontend.processing as processing
import frontend.razorpay_oauth as razorpay_oauth
import backend.src.razorpay_enforcement as enforcement

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def open_demo(app):
    assert not any(button.label == "Connect mock account" for button in app.button)
    return app


def navigate(app, view):
    return next(radio for radio in app.radio if radio.label == "Navigation").set_value(view).run()


def test_dashboard_opens_directly_in_demo_mode(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)

    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()

    assert not app.exception
    assert any("Demo mode" in markdown.value for markdown in app.markdown)
    assert any("Review queue" in markdown.value for markdown in app.markdown)
    assert app.dataframe
    assert app.metric
    assert not any("Connect a Razorpay account" in markdown.value for markdown in app.markdown)


def test_dashboard_can_require_real_oauth_when_mock_mode_is_disabled(monkeypatch):
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "false")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)

    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()

    assert not app.exception
    assert any("Connect a Razorpay account" in markdown.value for markdown in app.markdown)
    assert any("RAZORPAY_CLIENT_ID" in warning.value for warning in app.warning)
    assert not app.dataframe


def test_demo_dashboard_opens_transaction_dashboard_without_credentials(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)

    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()
    open_demo(app)

    assert not app.exception
    assert any("Demo mode" in markdown.value for markdown in app.markdown)
    assert any("Review queue" in markdown.value for markdown in app.markdown)
    assert any("Simulation only" in markdown.value for markdown in app.markdown)
    assert any("Demo every case" in markdown.value for markdown in app.markdown)
    assert any("Demo boundary" in markdown.value for markdown in app.markdown)
    demo_tables = [frame.value for frame in app.dataframe if "Case" in frame.value.columns]
    assert len(demo_tables) == 1
    assert demo_tables[0]["Case"].tolist() == [
        "Low risk / allowed", "Review band", "High risk / report", "False positive", "False negative"
    ]

    navigate(app, "Transactions")
    assert not app.exception
    assert len(app.dataframe) == 1
    assert len(app.dataframe[0].value) == 80
    assert list(app.dataframe[0].value.columns) == [
        "Txn", "Amount", "Method", "Velocity", "IP/Billing", "Device", "Score",
        "Risk", "Payment status",
    ]
    assert {"Transactions", "Captured", "Failed", "Captured value"} == {
        metric.label for metric in app.metric
    }
    report_buttons = [
        button for button in app.button if button.label == "Generate full evidence report"
    ]
    assert len(report_buttons) == 1

    navigate(app, "Model insights")
    assert not app.exception
    assert any("Held-out performance" in markdown.value for markdown in app.markdown)
    assert any("Cost-of-fraud policy explorer" in markdown.value for markdown in app.markdown)
    assert any("What drives the model's score" in markdown.value for markdown in app.markdown)

    navigate(app, "Ask about a payment")
    assert len(app.chat_input) == 1

    navigate(app, "Score a transaction")
    assert not app.exception
    assert any("Enter raw payment fields" in markdown.value for markdown in app.markdown)
    assert any(button.label == "Run fraud model" for button in app.button)
    assert {field.label for field in app.text_input} >= {
        "Transaction ID", "Timestamp (UTC)", "User ID", "Device ID", "Card ID",
        "Billing country", "IP country", "Merchant category",
    }


def test_mock_enforcement_capture_is_session_only_and_makes_no_http_calls(monkeypatch):
    outbound_calls = []

    def reject_network(self, method, url, *args, **kwargs):
        outbound_calls.append((method, url))
        raise AssertionError("mock enforcement must not make an HTTP request")

    monkeypatch.setattr(requests.sessions.Session, "request", reject_network)
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)
    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()
    open_demo(app)

    assert any(
        "Simulation only — no Razorpay API call or real money movement" in markdown.value
        for markdown in app.markdown
    )
    next(button for button in app.button if button.label == "Approve & capture").click().run()

    assert not app.exception
    assert outbound_calls == []
    assert app.session_state["mock_enforcement_scenarios"]["authorized"]["status"] == "Captured"
    audit = app.session_state["mock_enforcement_audit"]
    assert audit[-1]["Reviewer"] == "Demo Reviewer"
    assert audit[-1]["Action"] == "Approve & capture"
    assert audit[-1]["Resulting status"] == "Captured"


def test_mock_fraud_confirmation_stops_fulfillment_and_explains_auto_refund(monkeypatch):
    outbound_calls = []
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        lambda self, method, url, *args, **kwargs: outbound_calls.append((method, url)),
    )
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)
    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()
    open_demo(app)

    next(
        button for button in app.button
        if button.label == "Confirm fraud & release authorization"
    ).click().run()

    scenario = app.session_state["mock_enforcement_scenarios"]["authorized"]
    assert scenario["status"] == "Capture withheld"
    assert scenario["fulfillment_status"] == "Stopped"
    assert outbound_calls == []
    assert any("automatic refund" in warning.value for warning in app.warning)


def test_mock_captured_edge_case_refunds_only_in_session_state(monkeypatch):
    outbound_calls = []
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        lambda self, method, url, *args, **kwargs: outbound_calls.append((method, url)),
    )
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)
    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()
    open_demo(app)
    next(radio for radio in app.radio if radio.label == "Demo enforcement scenario").set_value(
        "captured"
    ).run()

    next(button for button in app.button if button.label == "Refund & stop fulfillment").click().run()

    scenario = app.session_state["mock_enforcement_scenarios"]["captured"]
    assert scenario["status"] == "Refunded"
    assert scenario["fulfillment_status"] == "Stopped"
    assert outbound_calls == []
    assert app.session_state["mock_enforcement_audit"][-1]["Action"] == "Refund & stop fulfillment"


def test_connected_dashboard_loads_and_filters_razorpay_transactions(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "true")

    payments = [
        {
            "id": "pay_captured",
            "created_at": 1_787_862_400,
            "amount": 12_500,
            "currency": "INR",
            "status": "captured",
            "method": "upi",
            "order_id": "order_one",
            "email": "buyer@example.com",
            "contact": "+919999999999",
            "international": False,
        },
        {
            "id": "pay_failed",
            "created_at": 1_787_776_000,
            "amount": 5_000,
            "currency": "INR",
            "status": "failed",
            "method": "card",
            "order_id": "order_two",
            "email": "second@example.com",
            "contact": "+918888888888",
            "international": True,
        },
    ]
    monkeypatch.setattr(razorpay_oauth, "fetch_payments", lambda *_args, **_kwargs: payments)

    app = AppTest.from_file("frontend/app.py", default_timeout=90)
    app.session_state["razorpay_connection"] = {
        "access_token": "test-access-token",
        "razorpay_account_id": "acc_test",
        "mode": "test",
    }
    app.run()
    navigate(app, "Transactions")

    assert not app.exception
    assert any("Account transactions" in markdown.value for markdown in app.markdown)
    assert any(
        "Razorpay does not supply device history" in info.value
        for info in app.info
    )
    assert not app.tabs
    assert not app.get("file_uploader")
    assert [date_input.label for date_input in app.date_input] == ["Transaction date range (UTC)"]
    assert {widget.label for widget in app.multiselect} == {"Status", "Payment method", "Currency"}
    assert [widget.label for widget in app.text_input] == ["Search"]
    assert [widget.label for widget in app.checkbox] == ["International payments only"]
    assert {metric.label: metric.value for metric in app.metric} == {
        "Transactions": "2",
        "Captured": "1",
        "Failed": "1",
        "Captured value": "₹125.00",
    }
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value["payment_id"].tolist() == ["pay_captured", "pay_failed"]
    assert "risk_score" not in app.dataframe[0].value.columns
    assert len(app.chat_input) == 0
    assert not any(button.label == "Generate full evidence report" for button in app.button)
    assert not any("Upload your CSV" in markdown.value for markdown in app.markdown)
    assert not any("Guided walkthrough" in markdown.value for markdown in app.markdown)

    next(widget for widget in app.multiselect if widget.label == "Status").select("failed").run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Transactions"] == "1"
    assert app.dataframe[0].value["payment_id"].tolist() == ["pay_failed"]


def test_connected_dashboard_converts_zero_decimal_currency(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "true")
    monkeypatch.setattr(
        razorpay_oauth,
        "fetch_payments",
        lambda *_args, **_kwargs: [{
            "id": "pay_jpy",
            "created_at": 1_787_862_400,
            "amount": 12_500,
            "currency": "JPY",
            "status": "captured",
            "method": "card",
        }],
    )
    app = AppTest.from_file("frontend/app.py", default_timeout=90)
    app.session_state["razorpay_connection"] = {
        "access_token": "test-access-token",
        "razorpay_account_id": "acc_test",
        "mode": "test",
    }

    app.run()
    navigate(app, "Transactions")

    assert not app.exception
    assert app.dataframe[0].value.iloc[0]["amount"] == 12_500.0


def test_pending_real_payment_requires_human_identity_and_click_to_capture(monkeypatch, tmp_path):
    database = tmp_path / "enforcement.sqlite3"
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "true")
    monkeypatch.setenv("RAZORPAY_MODE", "test")
    monkeypatch.setenv("RAZORPAY_ENFORCEMENT_DB", str(database))
    payment = {
        "id": "pay_pending_review",
        "order_id": "order_pending_review",
        "created_at": 1_787_862_400,
        "amount": 12_500,
        "currency": "INR",
        "status": "authorized",
        "captured": False,
        "method": "card",
    }
    store = enforcement.ReviewStore(database)
    store.process_event(
        "event_pending_review",
        {"event": "payment.authorized", "payload": {"payment": {"entity": payment}}},
    )
    gateway = type("Gateway", (), {
        "capture_calls": [],
        "fetch_payment": lambda self, payment_id: dict(payment),
        "fetch_order": lambda self, order_id: {
            "id": order_id, "amount": payment["amount"], "currency": payment["currency"],
        },
        "capture_payment": lambda self, payment_id, *, amount, currency: (
            self.capture_calls.append((payment_id, amount, currency))
            or {"id": payment_id, "status": "captured", "captured": True}
        ),
    })()
    monkeypatch.setattr(enforcement, "RazorpayGateway", lambda _token: gateway)
    monkeypatch.setattr(razorpay_oauth, "fetch_payments", lambda *_args, **_kwargs: [payment])
    app = AppTest.from_file("frontend/app.py", default_timeout=90)
    app.session_state["razorpay_connection"] = {
        "access_token": "test-access-token",
        "razorpay_account_id": "acc_test",
        "mode": "test",
    }

    app.run()

    assert not app.exception
    assert any("Human-approved Test Mode enforcement" in value.value for value in app.markdown)
    assert any(button.label == "Approve & capture" for button in app.button)
    assert any(button.label == "Confirm fraud & release authorization" for button in app.button)
    next(widget for widget in app.text_input if widget.label == "Reviewer identity").set_value(
        "reviewer@example.com"
    )
    next(button for button in app.button if button.label == "Approve & capture").click().run()

    assert not app.exception
    assert gateway.capture_calls == [("pay_pending_review", 12500, "INR")]
    assert enforcement.ReviewStore(database).get_review("pay_pending_review")["review_status"] == "approved_captured"


def test_already_captured_review_offers_human_triggered_full_refund(monkeypatch, tmp_path):
    database = tmp_path / "enforcement.sqlite3"
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "true")
    monkeypatch.setenv("RAZORPAY_MODE", "test")
    monkeypatch.setenv("RAZORPAY_ENFORCEMENT_DB", str(database))
    payment = {
        "id": "pay_refund_review",
        "order_id": "order_refund_review",
        "created_at": 1_787_862_400,
        "amount": 8_000,
        "amount_refunded": 0,
        "currency": "INR",
        "status": "captured",
        "captured": True,
        "method": "card",
    }
    store = enforcement.ReviewStore(database)
    store.process_event(
        "event_refund_review",
        {"event": "payment.authorized", "payload": {"payment": {"entity": payment}}},
    )
    gateway = type("Gateway", (), {
        "refund_calls": [],
        "fetch_payment": lambda self, payment_id: dict(payment),
        "refund_payment": lambda self, payment_id, *, idempotency_key: (
            self.refund_calls.append((payment_id, idempotency_key))
            or {"id": "rfnd_test", "status": "processed"}
        ),
    })()
    monkeypatch.setattr(enforcement, "RazorpayGateway", lambda _token: gateway)
    monkeypatch.setattr(razorpay_oauth, "fetch_payments", lambda *_args, **_kwargs: [payment])
    app = AppTest.from_file("frontend/app.py", default_timeout=90)
    app.session_state["razorpay_connection"] = {
        "access_token": "test-access-token",
        "razorpay_account_id": "acc_test",
        "mode": "test",
    }

    app.run()
    next(widget for widget in app.text_input if widget.label == "Reviewer identity").set_value(
        "reviewer@example.com"
    )
    next(button for button in app.button if button.label == "Refund & stop fulfillment").click().run()

    assert not app.exception
    assert len(gateway.refund_calls) == 1
    assert enforcement.ReviewStore(database).get_review("pay_refund_review")["review_status"] == "refunded"


def test_disconnect_revokes_razorpay_access_before_clearing_session(monkeypatch):
    revoked = []
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_CLIENT_ID", "client_123")
    monkeypatch.setenv("RAZORPAY_CLIENT_SECRET", "server-secret")
    monkeypatch.setenv("RAZORPAY_REDIRECT_URI", "https://app.test/callback")
    monkeypatch.setattr(
        razorpay_oauth,
        "revoke_access_token",
        lambda token, **kwargs: revoked.append((token, kwargs)),
    )
    monkeypatch.setattr(
        razorpay_oauth,
        "fetch_payments",
        lambda *_args, **_kwargs: [{
            "id": "pay_one",
            "created_at": 1_787_862_400,
            "amount": 12_500,
            "currency": "INR",
            "status": "captured",
            "method": "upi",
        }],
    )
    app = AppTest.from_file("frontend/app.py", default_timeout=90)
    app.session_state["razorpay_connection"] = {
        "access_token": "access-token",
        "razorpay_account_id": "acc_test",
    }
    app.run()

    next(button for button in app.button if button.label == "Disconnect").click().run()

    assert not app.exception
    assert revoked == [(
        "access-token",
        {"client_id": "client_123", "client_secret": "server-secret"},
    )]
    assert app.session_state["razorpay_connection"]["mock"] is True


def test_mock_report_is_generated_and_rendered_as_the_walkthrough_highlight(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)
    monkeypatch.setattr(
        processing,
        "generate_demo_transaction_report",
        lambda transaction, _url, **_kwargs: {
            "status": "generated",
            "summary": "Recent activity was elevated and this was a new device.",
            "evidence": [
                {"signal": "transaction_velocity", "detail": "17 recent transactions."},
                {"signal": "device_history", "detail": "Device was not previously seen."},
            ],
            "confidence_note": "Synthetic demonstration evidence for human review.",
            "recommended_action": "demo-block",
        },
    )
    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()
    open_demo(app)
    navigate(app, "Transactions")

    next(
        button for button in app.button if button.label == "Generate full evidence report"
    ).click().run()

    assert not app.exception
    assert any(
        "Recent activity was elevated and this was a new device." in info.value
        for info in app.info
    )


def test_mock_demo_shortcuts_route_to_each_review_surface(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)

    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()
    open_demo(app)

    next(button for button in app.button if button.label == "Open evidence report").click().run()
    assert app.session_state["fraudlens_view"] == "Transactions"
    assert app.session_state["demo_report_payment_id"]

    navigate(app, "Review queue")
    next(button for button in app.button if button.label == "Open policy audit").click().run()
    assert app.session_state["fraudlens_view"] == "Model insights"

    navigate(app, "Review queue")
    next(button for button in app.button if button.label == "Ask AI about a payment").click().run()
    assert app.session_state["fraudlens_view"] == "Ask about a payment"
