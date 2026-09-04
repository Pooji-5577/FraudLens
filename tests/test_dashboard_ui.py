import pytest
from streamlit.testing.v1 import AppTest

import dashboard.processing as processing
import dashboard.razorpay_oauth as razorpay_oauth

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_dashboard_is_hidden_until_razorpay_is_connected(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)

    app = AppTest.from_file("dashboard/app.py", default_timeout=90).run()

    assert not app.exception
    assert any("Connect your Razorpay account" in markdown.value for markdown in app.markdown)
    assert not app.dataframe
    assert not app.metric


def test_mock_connection_opens_transaction_dashboard_without_credentials(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)

    app = AppTest.from_file("dashboard/app.py", default_timeout=90).run()
    next(
        button for button in app.button if button.label == "Connect mock Razorpay account"
    ).click().run()

    assert not app.exception
    assert any(success.value == "Mock Razorpay account connected" for success in app.success)
    assert any("generated examples" in warning.value for warning in app.warning)
    assert any("SYNTHETIC RISK DEMO" in markdown.value for markdown in app.markdown)
    assert any("Synthetic scoring walkthrough" in markdown.value for markdown in app.markdown)
    assert any(
        "The score starts the review; the evidence report explains the decision"
        in markdown.value
        for markdown in app.markdown
    )
    assert len(app.dataframe) == 3
    assert len(app.dataframe[0].value) == 80
    assert list(app.dataframe[0].value.columns) == [
        "Txn", "Amount", "Velocity", "IP/billing", "Device", "Amt. dev.",
        "Hour", "Score", "Risk status", "Payment status", "Actual",
    ]
    assert any(
        "What influences the demo risk score" in markdown.value
        for markdown in app.markdown
    )
    assert {
        "Transactions", "Captured", "Failed", "Captured value", "Test transactions",
        "Flagged as fraud", "Correctly caught", "Precision", "Recall",
        "False-positive cost (legitimate payments blocked)",
        "False-negative cost (fraud that slipped through)",
        "Cheapest threshold", "Precision there", "Recall there", "Total held-out cost",
    } == {metric.label for metric in app.metric}
    assert any("Risk evaluation evidence" in markdown.value for markdown in app.markdown)
    assert any("Error breakdown" in markdown.value for markdown in app.markdown)
    assert any("Audit trail" in markdown.value for markdown in app.markdown)
    assert any("Cost-of-fraud policy explorer" in markdown.value for markdown in app.markdown)
    assert any("Defense-only" in markdown.value for markdown in app.markdown)
    report_buttons = [
        button for button in app.button if button.label == "✨ Generate full evidence report"
    ]
    assert len(report_buttons) == 1


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

    app = AppTest.from_file("dashboard/app.py", default_timeout=90)
    app.session_state["razorpay_connection"] = {
        "access_token": "test-access-token",
        "razorpay_account_id": "acc_test",
    }
    app.run()

    assert not app.exception
    assert any("RAZORPAY PAYMENT HISTORY" in markdown.value for markdown in app.markdown)
    assert any("Razorpay payment history" in markdown.value for markdown in app.markdown)
    assert any(
        "Real Razorpay payments don't yet include the signals" in info.value
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
        "Captured value": "125.00",
    }
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value["payment_id"].tolist() == ["pay_captured", "pay_failed"]
    assert "risk_score" not in app.dataframe[0].value.columns
    assert len(app.chat_input) == 1
    assert [widget.label for widget in app.selectbox] == ["Transaction for AI chat"]
    assert not any(button.label == "✨ Generate full evidence report" for button in app.button)
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
    app = AppTest.from_file("dashboard/app.py", default_timeout=90)
    app.session_state["razorpay_connection"] = {
        "access_token": "test-access-token",
        "razorpay_account_id": "acc_test",
    }

    app.run()

    assert not app.exception
    assert app.dataframe[0].value.iloc[0]["amount"] == 12_500.0


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
    app = AppTest.from_file("dashboard/app.py", default_timeout=90)
    app.session_state["razorpay_connection"] = {
        "access_token": "access-token",
        "razorpay_account_id": "acc_test",
    }
    app.run()

    next(button for button in app.button if button.label == "Disconnect account").click().run()

    assert not app.exception
    assert revoked == [(
        "access-token",
        {"client_id": "client_123", "client_secret": "server-secret"},
    )]
    assert "razorpay_connection" not in app.session_state


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
    app = AppTest.from_file("dashboard/app.py", default_timeout=90).run()
    next(
        button for button in app.button if button.label == "Connect mock Razorpay account"
    ).click().run()

    next(
        button for button in app.button if button.label == "✨ Generate full evidence report"
    ).click().run()

    assert not app.exception
    assert any("AI explanation" in markdown.value for markdown in app.markdown)
    assert any(
        "Recent activity was elevated and this was a new device." in info.value
        for info in app.info
    )
    assert any("Verified evidence" in markdown.value for markdown in app.markdown)
