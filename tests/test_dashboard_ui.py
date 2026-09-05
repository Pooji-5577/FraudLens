import io
from pathlib import Path

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

import frontend.processing as processing
import frontend.razorpay_oauth as razorpay_oauth

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def open_demo(app):
    assert not any(button.label == "Connect mock account" for button in app.button)
    return app


def navigate(app, view):
    return next(radio for radio in app.radio if radio.label == "Navigation").set_value(view).run()


def test_investigation_hidden_controls_do_not_clip_the_page_container():
    css = Path("frontend/redesign.css").read_text()

    assert 'div[data-testid="stVerticalBlock"]:has(.ti-hidden-controls)' not in css
    assert (
        'div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] '
        '.ti-hidden-controls)'
    ) in css


def test_case_metric_icons_use_material_ligatures_inside_their_circle():
    css = Path("frontend/redesign.css").read_text()
    icon_rule = css.split(".cm-metric-icon {", 1)[1].split("}", 1)[0]

    assert "font-feature-settings: 'liga'" in icon_rule
    assert "-webkit-font-feature-settings: 'liga'" in icon_rule
    assert ".cm-metric-icon" in css


def test_case_metric_text_styles_do_not_override_icon_layout():
    css = Path("frontend/redesign.css").read_text()

    assert ".cm-metric span {" not in css
    assert ".cm-metric > div > span {" in css


def test_transaction_metric_cards_share_a_responsive_grid_row():
    css = Path("frontend/redesign.css").read_text()
    metric_rule = css.split(
        'div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {',
        1,
    )[1].split("}", 1)[0]

    assert "display: grid" in metric_rule
    assert "grid-template-columns: repeat(4, minmax(0, 1fr))" in metric_rule
    assert "align-items: stretch" in metric_rule


def test_explorer_layout_is_bounded_and_signal_rows_can_shrink():
    css = Path("frontend/redesign.css").read_text()

    block_rule = css.split(".block-container {", 1)[1].split("}", 1)[0]
    explorer_rule = css.split(
        ".stApp:has(.transaction-explorer-route-label) .block-container {", 1
    )[1].split("}", 1)[0]
    importance_rule = css.split(".importance-row {", 1)[1].split("}", 1)[0]

    assert "box-sizing: border-box" in block_rule
    assert "width: 100%" in block_rule
    assert "max-width: min(1540px, 100%)" in explorer_rule
    assert "minmax(0, 1fr)" in importance_rule
    assert "@media (max-width: 1100px)" in css


def test_dashboard_opens_directly_in_demo_mode(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)

    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()

    assert not app.exception
    assert any("Fraud Spike Detection" in markdown.value for markdown in app.markdown)
    assert any("High-Risk Transactions" in markdown.value for markdown in app.markdown)
    assert not any("Review queue" in radio.options for radio in app.radio)
    assert not any("Demo every case in one session" in markdown.value for markdown in app.markdown)
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
    assert any("Fraud Spike Detection" in markdown.value for markdown in app.markdown)
    assert any("Transaction Trend" in markdown.value for markdown in app.markdown)
    assert any("Fraud Spike Detected!" in markdown.value for markdown in app.markdown)

    navigate(app, "Transaction explorer")
    assert not app.exception
    assert len(app.dataframe) == 1
    explorer_table = app.dataframe[0].value
    explorer_table = getattr(explorer_table, "data", explorer_table)
    assert len(explorer_table) == 80
    assert list(explorer_table.columns) == [
        "Transaction ID", "Date & Time", "Customer", "Amount", "Payment method",
        "Risk score", "Status", "Payment status",
    ]
    assert set(explorer_table["Status"].unique()) <= {"Flagged", "Review", "Legit"}
    metric_labels = " ".join(metric.label for metric in app.metric)
    for expected in ("Total transactions", "Flagged transactions", "Total transaction amount", "Avg. transaction amount"):
        assert expected in metric_labels
    assert not any(button.label == "Generate full evidence report" for button in app.button)

    navigate(app, "Overview")
    assert not app.exception
    assert any("Fraud Spike Detection" in markdown.value for markdown in app.markdown)
    assert any("Detect Spikes" in markdown.value for markdown in app.markdown)
    assert any("High-Risk Transactions" in markdown.value for markdown in app.markdown)

    navigate(app, "Transaction investigation")
    assert not app.exception
    assert any(button.label == "Generate full evidence report" for button in app.button)
    assert not app.chat_input
    assert any(widget.label == "Ask about this transaction" for widget in app.text_input)
    assert any(button.label == "Send question" for button in app.button)
    assert any(button.label == "Previous transaction" for button in app.button)
    assert any(button.label == "Next transaction" for button in app.button)
    assert not any(
        "This transaction is flagged due to several high-risk indicators" in markdown.value
        for markdown in app.markdown
    )

    navigate(app, "Case management")
    assert not app.exception
    assert any("Case Management" in markdown.value for markdown in app.markdown)


def test_transaction_investigation_submits_questions_to_the_backend_client(monkeypatch):
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")
    monkeypatch.setattr(
        processing,
        "ask_preview_transaction_question",
        lambda transaction, question, history, _url: {
            "status": "generated",
            "transaction_id": transaction["transaction_id"],
            "answer": f"Backend answer for: {question}",
            "provider": "grounded-rules",
        },
    )
    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()
    navigate(app, "Transaction investigation")

    question = next(widget for widget in app.text_input if widget.label == "Ask about this transaction")
    question.set_value("Why was this flagged?").run()
    next(button for button in app.button if button.label == "Send question").click().run()

    assert not app.exception
    assert any("Backend answer for: Why was this flagged?" in markdown.value for markdown in app.markdown)


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
    navigate(app, "Transaction explorer")

    assert not app.exception
    assert any("Account transactions" in markdown.value for markdown in app.markdown)
    assert any(
        "Razorpay does not supply device history" in info.value
        for info in app.info
    )
    assert not app.tabs
    assert len(app.get("file_uploader")) == 1
    assert app.get("file_uploader")[0].label == "Attach transactions CSV"
    assert any(button.label == "Run model" for button in app.button)
    assert [date_input.label for date_input in app.date_input] == ["Transaction date range (UTC)"]
    assert {widget.label for widget in app.multiselect} == {"Status", "Payment method", "Currency"}
    assert [widget.label for widget in app.text_input] == ["Search", "Search"]
    assert [widget.label for widget in app.checkbox] == ["International payments only"]

    def _metric_value(fragment):
        return next(metric.value for metric in app.metric if fragment in metric.label)

    assert _metric_value("Total transactions") == "2"
    assert _metric_value("Captured") == "1"
    assert _metric_value("Total transaction amount") == "₹175.00"
    assert _metric_value("Avg. transaction amount") == "₹87.50"
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value["payment_id"].tolist() == ["pay_captured", "pay_failed"]
    assert "risk_score" not in app.dataframe[0].value.columns
    assert len(app.chat_input) == 0
    assert not any(button.label == "Generate full evidence report" for button in app.button)
    assert not any("Upload your CSV" in markdown.value for markdown in app.markdown)
    assert not any("Guided walkthrough" in markdown.value for markdown in app.markdown)

    next(widget for widget in app.multiselect if widget.label == "Status").select("failed").run()
    assert not app.exception
    assert _metric_value("Total transactions") == "1"
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
    navigate(app, "Transaction explorer")

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
    app = AppTest.from_file("frontend/app.py", default_timeout=90)
    app.session_state["razorpay_connection"] = {
        "access_token": "access-token",
        "razorpay_account_id": "acc_test",
    }
    app.run()
    navigate(app, "Overview")

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
    navigate(app, "Transaction investigation")

    next(
        button for button in app.button if button.label == "Generate full evidence report"
    ).click().run()

    assert not app.exception
    assert any(
        "Recent activity was elevated and this was a new device." in info.value
        for info in app.info
    )


def test_transaction_explorer_offers_a_csv_batch_tester(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")
    monkeypatch.delenv("RAZORPAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("RAZORPAY_REDIRECT_URI", raising=False)

    app = AppTest.from_file("frontend/app.py", default_timeout=90).run()
    open_demo(app)
    navigate(app, "Transaction explorer")

    assert not app.exception
    assert not app.get("expander")
    assert any("Run the fraud model on a CSV" in markdown.value for markdown in app.markdown)
    uploaders = app.get("file_uploader")
    assert len(uploaders) == 1
    assert uploaders[0].label == "Attach transactions CSV"
    assert any(button.label == "Run model" for button in app.button)
    assert any(button.label == "Download CSV template" for button in app.get("download_button"))
    assert any(
        "transaction_id, timestamp, user_id, device_id, card_id, amount, billing_country, "
        "ip_country, merchant_category" in caption.value
        for caption in app.caption
    )


def test_selecting_a_new_csv_does_not_keep_the_previous_signal_importance(monkeypatch):
    monkeypatch.setenv("RAZORPAY_AUTH_DISABLED", "false")
    monkeypatch.setenv("RAZORPAY_MOCK_AUTH", "true")

    def fake_score(transactions, filename, _api_url):
        scored = transactions.copy()
        scored["score"] = 0.11
        scored["flagged"] = False
        scored["blocked"] = False
        scored["reasons"] = [["test reason"] for _ in range(len(scored))]
        importance = {
            "first.csv": {"Amount deviation": 90.0, "New device": 10.0},
            "second.csv": {"Geography mismatch": 80.0, "Time of day": 20.0},
        }[filename]
        support = {
            "first.csv": {"Amount deviation": 80.0, "New device": 20.0},
            "second.csv": {"Geography mismatch": 70.0, "Time of day": 30.0},
        }[filename]
        return {
            "dataset_id": filename,
            "filename": filename,
            "row_count": len(scored),
            "scored": scored,
            "storage_status": "saved",
            "storage_error": None,
            "signal_importance_percent": importance,
            "signal_support_percent": support,
            "decision_threshold": 0.23,
        }

    monkeypatch.setattr(processing, "score_and_save_uploaded_dataset", fake_score)
    selected_upload = {"value": None}

    def fake_file_uploader(*_args, **_kwargs):
        upload = selected_upload["value"]
        if upload is not None:
            upload.seek(0)
        return upload

    monkeypatch.setattr(streamlit, "file_uploader", fake_file_uploader)
    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=90).run()
    navigate(app, "Transaction explorer")

    class TestUploadedFile(io.BytesIO):
        def __init__(self, name, content):
            super().__init__(content)
            self.name = name
            self.file_id = name

    def select_csv(name, content):
        selected_upload["value"] = TestUploadedFile(name, content)
        app.session_state["csv_tester_upload"] = selected_upload["value"]
        return app.run()

    first_csv = (
        b"transaction_id,timestamp,user_id,device_id,card_id,amount,billing_country,"
        b"ip_country,merchant_category\n"
        b"txn-a,2026-01-15T09:30:00Z,user-a,device-a,card-a,100,IN,IN,grocery\n"
    )
    select_csv("first.csv", first_csv)
    assert not any(
        '<div class="importance-panel">' in markdown.value
        for markdown in app.markdown
    )
    next(button for button in app.button if button.label == "Run model").click().run()

    first_panel = next(
        markdown.value
        for markdown in app.markdown
        if '<div class="importance-panel">' in markdown.value
    )
    assert "90.0%" in first_panel
    assert any("Top model driver" in markdown.value for markdown in app.markdown)
    assert any("Contributed in 80.0%" in markdown.value for markdown in app.markdown)
    assert any("first.csv" in caption.value for caption in app.caption)

    second_csv = first_csv.replace(b"txn-a", b"txn-b").replace(b"100,IN,IN", b"9999,IN,US")
    select_csv("second.csv", second_csv)

    assert any(
        "Run the model to calculate signal influence for this CSV" in info.value
        for info in app.info
    )
    assert not any(
        '<div class="importance-panel">' in markdown.value and "90.0%" in markdown.value
        for markdown in app.markdown
    )

    next(button for button in app.button if button.label == "Run model").click().run()
    second_panel = next(
        markdown.value
        for markdown in app.markdown
        if '<div class="importance-panel">' in markdown.value
    )
    assert "80.0%" in second_panel
    assert "90.0%" not in second_panel
    assert any("second.csv" in caption.value for caption in app.caption)
