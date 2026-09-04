"""FraudLens transaction review dashboard."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from frontend.processing import (
    ScoringAPIError,
    add_fraud_case_note,
    ask_preview_transaction_question,
    cheapest_threshold_row,
    cost_curve_for_ratio,
    csv_injection_safe,
    demo_case_catalog,
    generate_demo_transaction_report,
    get_fraud_case,
    list_fraud_cases,
    load_global_importance,
    load_threshold_curve,
    risk_audit_rows,
    risk_evidence_summary,
    set_fraud_case_status,
)
from frontend.mock_enforcement import apply_mock_action, initial_mock_scenarios
from frontend.razorpay_oauth import (
    RazorpayOAuthError,
    amount_from_subunits,
    build_authorization_url,
    consume_oauth_state,
    exchange_authorization_code,
    fetch_payments,
    issue_oauth_state,
    revoke_access_token,
)
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="FraudLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUT_LENS_MARK_SVG = (
    '<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:100%;">'
    '<polygon points="60,26 89.4,43 89.4,77 60,94 30.6,77 30.6,43" '
    'fill="none" stroke="{stroke}" stroke-width="7" stroke-linejoin="round" />'
    '<polygon points="69,44.4 78,60 69,75.6 51,75.6 42,60 51,44.4" fill="#8ec5ff" />'
    '</svg>'
)


def brand_lockup(size: str = "26px", text_size: str = "1.05rem", text_color: str = "#eef4ff", stroke: str = "#5d94fc") -> str:
    mark = CUT_LENS_MARK_SVG.format(stroke=stroke)
    return (
        f'<div class="brand-lockup" style="font-size:{text_size};color:{text_color};">'
        f'<span class="brand-mark" style="width:{size};height:{size};">{mark}</span>'
        f'<span>Fraud<span style="color:{stroke};">Lens</span></span></div>'
    )


st.markdown(
    f"<style>{(PROJECT_ROOT / 'frontend' / 'redesign.css').read_text()}</style>",
    unsafe_allow_html=True,
)

RAZORPAY_CLIENT_ID = os.getenv("RAZORPAY_CLIENT_ID", "")
RAZORPAY_CLIENT_SECRET = os.getenv("RAZORPAY_CLIENT_SECRET", "")
RAZORPAY_REDIRECT_URI = os.getenv("RAZORPAY_REDIRECT_URI", "")
RAZORPAY_MODE = os.getenv("RAZORPAY_MODE", "test").lower()
RAZORPAY_ENFORCEMENT_DB = Path(
    os.getenv(
        "RAZORPAY_ENFORCEMENT_DB",
        str(PROJECT_ROOT / "backend" / "data" / "razorpay_enforcement.sqlite3"),
    )
)
SCORING_API_URL = os.getenv("SCORING_API_URL", "http://localhost:8000").rstrip("/")
DEMO_BLOCKING_THRESHOLD = float(os.getenv("DEMO_BLOCKING_THRESHOLD", "0.65"))
DEMO_REVIEW_THRESHOLD = float(os.getenv("DEMO_REVIEW_THRESHOLD", "0.40"))
RAZORPAY_MOCK_AUTH = os.getenv("RAZORPAY_MOCK_AUTH", "true").lower() == "true"
THRESHOLD_CURVE_PATH = PROJECT_ROOT / "backend" / "reports" / "metrics" / "threshold_curve.csv"
GLOBAL_IMPORTANCE_PATH = PROJECT_ROOT / "backend" / "reports" / "metrics" / "global_feature_importance.json"

CASE_STATUS_LABELS = {
    "open": "Open",
    "under_investigation": "Under investigation",
    "confirmed_fraud": "Confirmed fraud",
    "false_positive": "False positive",
}


def _signal_reasons(row: pd.Series) -> list[str]:
    """Human-readable reason chips derived from a mock row's own risk signals."""
    reasons = []
    velocity = row.get("velocity")
    if velocity is not None and not pd.isna(velocity) and int(velocity) >= 8:
        reasons.append(f"{int(velocity)} recent transactions")
    if row.get("ip_billing_mismatch"):
        reasons.append("IP/billing mismatch")
    if row.get("new_device"):
        reasons.append("New device")
    deviation = row.get("amount_deviation")
    if deviation is not None and not pd.isna(deviation) and abs(float(deviation)) >= 50:
        reasons.append(f"Amount {float(deviation):+.0f}% vs. baseline")
    return reasons or ["Elevated composite risk score"]


def oauth_is_configured() -> bool:
    """Return whether the server has a complete Razorpay Partner app config."""
    values = (RAZORPAY_CLIENT_ID, RAZORPAY_CLIENT_SECRET, RAZORPAY_REDIRECT_URI)
    return all(value and not value.startswith("replace-with-") for value in values)


def render_connection_screen() -> None:
    """Render the optional real-account OAuth entry point when mock mode is disabled."""
    callback_error = st.query_params.get("error")
    authorization_code = st.query_params.get("code")
    returned_state = st.query_params.get("state")
    if callback_error:
        st.error("Razorpay account access was not approved. Please try connecting again.")
    elif authorization_code:
        if not consume_oauth_state(returned_state):
            st.error("The Razorpay login response could not be verified. Please start again.")
        elif not oauth_is_configured():
            st.error("Razorpay OAuth is not configured on this server.")
        else:
            try:
                token = exchange_authorization_code(
                    authorization_code,
                    client_id=RAZORPAY_CLIENT_ID,
                    client_secret=RAZORPAY_CLIENT_SECRET,
                    redirect_uri=RAZORPAY_REDIRECT_URI,
                    mode=RAZORPAY_MODE,
                )
            except RazorpayOAuthError as exc:
                st.error(str(exc))
            else:
                st.session_state["razorpay_connection"] = token
                st.query_params.clear()
                st.rerun()

    st.markdown(
        '<div class="login-shell"><h1>Connect a Razorpay account</h1>'
        '<p>Read access loads Test Mode payment history for review. Write access is required '
        'for explicit human-approved capture and refund actions.</p></div>',
        unsafe_allow_html=True,
    )
    if RAZORPAY_MODE != "test":
        st.error("FraudLens Razorpay access is restricted to Test Mode.")
    elif oauth_is_configured():
        st.link_button(
            "Connect Razorpay Test Mode account",
            build_authorization_url(
                RAZORPAY_CLIENT_ID, RAZORPAY_REDIRECT_URI, issue_oauth_state()
            ),
            type="primary",
            use_container_width=True,
        )
        st.caption("Credentials stay server-side; this dashboard never receives the client secret.")
    else:
        st.warning(
            "Razorpay OAuth is not configured. Set RAZORPAY_CLIENT_ID, "
            "RAZORPAY_CLIENT_SECRET, and RAZORPAY_REDIRECT_URI, or leave "
            "RAZORPAY_MOCK_AUTH=true for the local demo."
        )
    st.stop()


def payments_frame(payments: list[dict]) -> pd.DataFrame:
    """Select merchant-facing fields instead of exposing the raw API payload."""
    if not payments:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "payment_id": [payment.get("id", "") for payment in payments],
            "created_at": [
                pd.to_datetime(payment.get("created_at"), unit="s", utc=True, errors="coerce")
                for payment in payments
            ],
            "amount": [
                amount_from_subunits(payment.get("amount", 0), payment.get("currency", ""))
                for payment in payments
            ],
            "currency": [payment.get("currency", "") for payment in payments],
            "status": [payment.get("status", "") for payment in payments],
            "method": [payment.get("method", "") for payment in payments],
            "order_id": [payment.get("order_id") or "" for payment in payments],
            "email": [payment.get("email") or "" for payment in payments],
            "contact": [payment.get("contact") or "" for payment in payments],
            "international": [bool(payment.get("international", False)) for payment in payments],
            "velocity": [payment.get("velocity") for payment in payments],
            "ip_billing_mismatch": [payment.get("ip_billing_mismatch") for payment in payments],
            "new_device": [payment.get("new_device") for payment in payments],
            "amount_deviation": [payment.get("amount_deviation") for payment in payments],
            "risk_score": [payment.get("risk_score") for payment in payments],
            "risk_status": [payment.get("risk_status") for payment in payments],
            "actual": [payment.get("actual") for payment in payments],
        }
    ).sort_values("created_at", ascending=False, ignore_index=True)


def mock_payments(from_timestamp: int, to_timestamp: int) -> list[dict]:
    """Create deterministic Razorpay-shaped transactions for the requested period."""
    start = pd.to_datetime(from_timestamp, unit="s", utc=True)
    end = pd.to_datetime(to_timestamp, unit="s", utc=True)
    if start > end:
        return []
    timestamps = pd.date_range(start=start, end=end, periods=80)
    statuses = ["captured", "captured", "captured", "failed", "authorized", "refunded"]
    methods = ["upi", "card", "netbanking", "wallet", "card", "upi"]
    currencies = ["INR", "INR", "INR", "INR", "USD", "INR"]
    payments = []
    for index, timestamp in enumerate(reversed(timestamps)):
        status = statuses[index % len(statuses)]
        velocity = 1 + ((index * 5) % 18)
        ip_billing_mismatch = index % 7 == 0
        new_device = index % 5 == 0
        amount_deviation = ((index * 19) % 141) - 20
        odd_hour = timestamp.hour < 6 or timestamp.hour >= 23
        risk_score = min(
            0.99,
            0.35 * min(velocity / 12, 1)
            + 0.20 * ip_billing_mismatch
            + 0.33 * new_device
            + 0.04 * min(abs(amount_deviation) / 100, 1)
            + 0.08 * odd_hour,
        )
        risk_status = (
            "High risk"
            if risk_score >= DEMO_BLOCKING_THRESHOLD
            else "Review"
            if risk_score >= DEMO_REVIEW_THRESHOLD
            else "Low risk"
        )
        payments.append(
            {
                "id": f"pay_demo_{index + 1:04d}",
                "created_at": int(timestamp.timestamp()),
                "amount": 9_900 + ((index * 7_913) % 240_000),
                "currency": currencies[index % len(currencies)],
                "status": status,
                "method": methods[index % len(methods)],
                "order_id": f"order_demo_{index + 1:04d}",
                "email": f"customer{(index % 18) + 1}@example.com",
                "contact": f"+9190000{index % 100000:05d}",
                "international": currencies[index % len(currencies)] != "INR",
                "velocity": velocity,
                "ip_billing_mismatch": ip_billing_mismatch,
                "new_device": new_device,
                "amount_deviation": amount_deviation,
                "risk_score": round(risk_score, 3),
                "risk_status": risk_status,
                "actual": "Fraud" if index % 11 == 0 else "Legitimate",
            }
        )
    return payments


def format_currency_costs(costs: dict[str, float]) -> str:
    symbols = {"INR": "₹", "USD": "$"}
    if not costs:
        return "₹0.00"
    return " · ".join(
        f"{symbols.get(currency, currency + ' ')}{amount:,.2f}"
        for currency, amount in sorted(costs.items())
    )


def render_enforcement_panel(connection: dict) -> None:
    """Render explicit human actions for webhook-created review records."""
    from backend.src.razorpay_enforcement import (
        EnforcementError,
        EnforcementService,
        RazorpayGateway,
    )
    from backend.src.review_store import SupabaseStoreError, review_store_from_environment

    st.markdown("### Human-approved Test Mode enforcement")
    if RAZORPAY_MODE != "test":
        st.error("Enforcement is disabled: FraudLens is restricted to Razorpay Test Mode.")
        return
    try:
        store = review_store_from_environment(RAZORPAY_ENFORCEMENT_DB)
        actionable = [
            review
            for review in store.list_reviews()
            if review["review_status"] in {"pending_review", "already_captured"}
        ]
    except SupabaseStoreError as exc:
        st.error(f"Review storage is unavailable: {exc}")
        return
    if not actionable:
        st.caption("No webhook-authorized payments are waiting for a human decision.")
        return
    if connection.get("mode") != "test":
        st.error(
            "Enforcement is disabled for this unmarked or legacy OAuth session. Reconnect using "
            "the Test Mode OAuth flow."
        )
        return

    selected_id = st.selectbox(
        "Payment awaiting review",
        [review["payment_id"] for review in actionable],
        key="enforcement_payment_id",
    )
    review = next(item for item in actionable if item["payment_id"] == selected_id)
    st.write(
        f"**{selected_id}** · {review['currency']} {amount_from_subunits(review['amount'], review['currency']):,.2f} "
        f"· Razorpay status: `{review['payment_status']}`"
    )
    if review.get("risk_score") is None:
        st.info(
            "No FraudLens score or SHAP evidence is available for this real payment because Razorpay "
            "does not supply the device, IP-geography, velocity, and history inputs required by the model."
        )
    else:
        st.metric("FraudLens score", f"{float(review['risk_score']):.3f}")
        if review.get("evidence"):
            st.json(review["evidence"])

    reviewer = st.text_input(
        "Reviewer identity",
        placeholder="Name or work email",
        key=f"reviewer_identity_{selected_id}",
    )
    try:
        service = EnforcementService(
            store,
            RazorpayGateway(connection["access_token"]),
            mode=RAZORPAY_MODE,
        )
        if review["review_status"] == "pending_review":
            approve_column, fraud_column = st.columns(2)
            with approve_column:
                approve = st.button("Approve & capture", type="primary", use_container_width=True)
            with fraud_column:
                confirm_fraud = st.button(
                    "Confirm fraud & release authorization", use_container_width=True
                )
            if approve:
                service.approve_and_capture(selected_id, actor=reviewer)
                st.success("Payment captured in Razorpay Test Mode; fulfillment approved.")
            if confirm_fraud:
                service.confirm_fraud(selected_id, actor=reviewer)
                st.success(
                    "Capture withheld and fulfillment stopped. Razorpay will auto-refund the uncaptured "
                    "authorization after the configured timeout."
                )
        else:
            st.warning("Already captured—this payment could not be held for review.")
            if st.button("Refund & stop fulfillment", type="primary", use_container_width=True):
                service.refund_and_stop(selected_id, actor=reviewer)
                st.success("Full refund requested in Razorpay Test Mode; fulfillment stopped.")
    except EnforcementError as exc:
        st.error(str(exc))

    audit = store.list_audit_entries(selected_id)
    if audit:
        st.markdown("#### Enforcement audit log")
        st.dataframe(pd.DataFrame(audit), use_container_width=True, hide_index=True)


def render_mock_enforcement_panel() -> None:
    """Render a session-only walkthrough that cannot reach Razorpay."""
    st.markdown(
        """
        <div class="mock-enforcement-shell">
          <div class="mock-enforcement-kicker">SIMULATED ENFORCEMENT WALKTHROUGH</div>
          <h2>Review queue</h2>
          <p>A payment the bank authorized but has not yet been captured. Review it before it settles.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sim-banner">Simulation only — no Razorpay API call or real money movement</div>',
        unsafe_allow_html=True,
    )
    scenarios = st.session_state.setdefault(
        "mock_enforcement_scenarios", initial_mock_scenarios()
    )
    audit_log = st.session_state.setdefault("mock_enforcement_audit", [])
    scenario_id = st.radio(
        "Demo enforcement scenario",
        ["authorized", "captured"],
        format_func=lambda value: scenarios[value]["scenario"],
        horizontal=True,
        key="mock_enforcement_scenario_id",
    )
    scenario = scenarios[scenario_id]
    amount = amount_from_subunits(scenario["amount"], scenario["currency"])
    symbol = {"INR": "₹", "USD": "$"}.get(scenario["currency"], scenario["currency"] + " ")
    st.markdown(
        f"""
        <div class="payment-card">
          <div class="payment-card-head">
            <div><div class="payment-id">{scenario['payment_id']}</div>
              <span class="status-badge">{scenario['status']}</span></div>
            <div><div class="payment-amount">{symbol}{amount:,.2f}</div>
              <div class="payment-subline">{scenario['method']} · {scenario['amount']} subunits</div></div>
          </div>
          <div class="payment-fields">
            <div><small>Fulfillment</small><strong>{scenario['fulfillment_status']}</strong></div>
            <div><small>Billing country</small><strong>{scenario['billing_country']}</strong></div>
            <div><small>Customer</small><strong>{scenario['billing_name']}</strong></div>
            <div><small>Email</small><strong>{scenario['billing_email']}</strong></div>
            <div><small>Contact</small><strong>{scenario['billing_contact']}</strong></div>
            <div><small>Order ID</small><strong class="payment-id">{scenario['order_id']}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Real Razorpay payment data does not include enough device history, IP geography, velocity, "
        "and customer history for a genuine FraudLens model score. This payment and decision are "
        "simulated for demonstration; no score has been fabricated."
    )

    if scenario_id == "authorized":
        approve_column, fraud_column = st.columns(2)
        with approve_column:
            approve = st.button(
                "Approve & capture",
                key="mock_approve_capture",
                type="primary",
                use_container_width=True,
                disabled=scenario["status"] != "Authorized — awaiting review",
            )
        with fraud_column:
            confirm_fraud = st.button(
                "Confirm fraud & release authorization",
                key="mock_confirm_fraud",
                use_container_width=True,
                disabled=scenario["status"] != "Authorized — awaiting review",
            )
        if approve:
            scenario = apply_mock_action(
                scenarios, audit_log, scenario_id="authorized", action="approve"
            )
            st.success("Simulation updated to Captured. No capture request was sent.")
        if confirm_fraud:
            scenario = apply_mock_action(
                scenarios, audit_log, scenario_id="authorized", action="confirm_fraud"
            )
            st.success("Capture withheld and simulated fulfillment stopped.")
    else:
        refund = st.button(
            "Refund & stop fulfillment",
            key="mock_refund_stop",
            type="primary",
            use_container_width=True,
            disabled=scenario["status"] != "Captured before review",
        )
        if refund:
            scenario = apply_mock_action(
                scenarios, audit_log, scenario_id="captured", action="refund"
            )
            st.success("Simulation updated to Refunded. No refund request was sent.")

    if scenario["status"] == "Capture withheld":
        st.warning(
            "In the real manual-capture flow, the uncaptured authorization would wait for Razorpay's "
            "automatic refund after the merchant's configured timeout. This simulation does not execute it."
        )
    st.markdown("### Audit log")
    if audit_log:
        st.dataframe(pd.DataFrame(audit_log), use_container_width=True, hide_index=True)
    else:
        st.caption("No actions taken yet in this session.")


def _open_demo_view(view: str, payment_id: str | None = None) -> None:
    """Route the mock tour buttons through the existing navigation rail."""
    st.session_state["fraudlens_view"] = view
    if payment_id:
        st.session_state["demo_report_payment_id"] = payment_id
        st.session_state["investigation_payment_id"] = payment_id


def render_mock_demo_guide(transactions: pd.DataFrame) -> None:
    """Show a deterministic, end-to-end tour of every mock-only demo case."""
    if transactions.empty:
        st.info("The mock demo has no transactions for the selected date range.")
        return
    catalog = demo_case_catalog(transactions, threshold=DEMO_BLOCKING_THRESHOLD)
    high_risk = transactions.loc[
        transactions["risk_score"].ge(DEMO_BLOCKING_THRESHOLD)
    ].sort_values("risk_score", ascending=False)
    report_payment_id = str(high_risk.iloc[0]["payment_id"]) if not high_risk.empty else None

    st.markdown(
        """
        <div class="demo-tour">
          <div class="demo-tour-kicker">COMPLETE MOCK DEMO</div>
          <h2>Demo every case in one session</h2>
          <p>These synthetic rows deliberately include the normal risk bands, both error types,
          a reviewer-ready evidence report, grounded AI chat, and the three human enforcement outcomes.
          Use the shortcuts below to walk a judge through the whole product without touching Razorpay.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Risk cases in this dataset")
    st.caption(
        "Counts and examples are derived from the same 80 synthetic rows shown in Transactions. "
        "The labels below are synthetic evaluation labels, not findings about a real customer."
    )
    metric_columns = st.columns(len(catalog))
    for column, row in zip(metric_columns, catalog.to_dict("records")):
        with column:
            column.metric(row["Case"], f"{int(row['Rows']):,}")
    st.dataframe(
        catalog,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rows": st.column_config.NumberColumn("Rows", format="%d"),
            "Example score": st.column_config.ProgressColumn(
                "Example score", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
    )

    st.markdown(
        '<div class="demo-boundary"><strong>Demo boundary:</strong> this workspace opens directly '
        'in synthetic demo mode. Real payments are intentionally not scored because the Payments API '
        'does not include the device, IP-geography, velocity, and history signals this trained model '
        'needs.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Shortcuts for the live walkthrough")
    shortcut_columns = st.columns(4)
    with shortcut_columns[0]:
        st.button(
            "Open fraud overview",
            key="demo_open_overview",
            use_container_width=True,
            on_click=_open_demo_view,
            args=("Fraud overview",),
        )
    with shortcut_columns[1]:
        st.button(
            "Investigate top alert",
            key="demo_open_investigation",
            type="primary",
            use_container_width=True,
            disabled=report_payment_id is None,
            on_click=_open_demo_view,
            args=("Transaction investigation", report_payment_id),
        )
    with shortcut_columns[2]:
        st.button(
            "Explore all transactions",
            key="demo_open_explorer",
            use_container_width=True,
            on_click=_open_demo_view,
            args=("Transaction explorer",),
        )
    with shortcut_columns[3]:
        st.button(
            "Open case management",
            key="demo_open_cases",
            use_container_width=True,
            on_click=_open_demo_view,
            args=("Case management",),
        )
    st.caption(
        "Review queue (this page) covers Authorized → Approve & capture, Authorized → Confirm fraud & "
        "release authorization, and Captured → Refund & stop fulfillment. Every action is session-only "
        "in the demo session and is recorded in the visible audit log."
    )


def render_page_header(title: str, copy: str = "") -> None:
    st.markdown(
        f'<div class="page-head"><h1>{title}</h1><p>{copy}</p></div>',
        unsafe_allow_html=True,
    )


def load_dashboard_transactions(*, show_date_filter: bool = False) -> pd.DataFrame:
    today = pd.Timestamp.now(tz="UTC").date()
    default_start = today - pd.Timedelta(days=29)
    loaded_range = st.session_state.get("razorpay_loaded_range", (default_start, today))
    selected_dates = loaded_range
    apply_dates = False
    if show_date_filter:
        with st.form("razorpay_date_filters"):
            date_column, action_column = st.columns([3, 1], vertical_alignment="bottom")
            with date_column:
                selected_dates = st.date_input(
                    "Transaction date range (UTC)", value=loaded_range, max_value=today
                )
            with action_column:
                apply_dates = st.form_submit_button(
                    "Apply date range", type="primary", use_container_width=True
                )
    if apply_dates or "razorpay_payments" not in st.session_state:
        if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
            st.warning("Choose both a start date and an end date.")
            return payments_frame(st.session_state.get("razorpay_payments", []))
        range_start, range_end = selected_dates
        if range_start > range_end:
            st.warning("The start date must be before the end date.")
            return payments_frame(st.session_state.get("razorpay_payments", []))
        from_timestamp = int(pd.Timestamp(range_start, tz="UTC").timestamp())
        to_timestamp = int(
            (pd.Timestamp(range_end, tz="UTC") + pd.Timedelta(days=1, seconds=-1)).timestamp()
        )
        connection = st.session_state["razorpay_connection"]
        try:
            with st.spinner("Loading transactions from Razorpay…"):
                st.session_state["razorpay_payments"] = (
                    mock_payments(from_timestamp, to_timestamp)
                    if connection.get("mock")
                    else fetch_payments(
                        connection["access_token"],
                        from_timestamp=from_timestamp,
                        to_timestamp=to_timestamp,
                    )
                )
                st.session_state["razorpay_loaded_range"] = (range_start, range_end)
        except RazorpayOAuthError as exc:
            st.error(str(exc))
    return payments_frame(st.session_state.get("razorpay_payments", []))


def render_transactions_view(transactions: pd.DataFrame, *, is_mock: bool) -> None:
    render_page_header(
        "Transaction explorer" if is_mock else "Account transactions",
        "Filterable by date, amount, status, geography, and device across every loaded transaction."
        if is_mock else "Payment history loaded from the connected Razorpay Test Mode account.",
    )
    if is_mock:
        st.info("Demo signals are simulated; they are not real Razorpay fraud findings.")
    one, two, three, four = st.columns(4)
    with one:
        search = st.text_input("Search", placeholder="Payment ID, order ID, email…")
        statuses = st.multiselect(
            "Status", sorted(value for value in transactions["status"].dropna().unique() if value)
        )
    with two:
        methods = st.multiselect(
            "Payment method", sorted(value for value in transactions["method"].dropna().unique() if value)
        )
        currencies = st.multiselect(
            "Currency", sorted(value for value in transactions["currency"].dropna().unique() if value)
        )
    with three:
        risk_statuses = []
        if transactions["risk_status"].notna().any():
            risk_statuses = st.multiselect(
                "Risk status",
                [v for v in ("High risk", "Review", "Low risk") if v in set(transactions["risk_status"])],
            )
        international_only = st.checkbox("International payments only")
    with four:
        geography_filter = "Any"
        device_filter = "Any"
        if is_mock:
            geography_filter = st.selectbox(
                "Geography", ["Any", "Match", "Mismatch"], key="explorer_geography_filter"
            )
            device_filter = st.selectbox(
                "Device", ["Any", "Known", "New"], key="explorer_device_filter"
            )
    filtered = transactions.copy()
    if search:
        needle = search.casefold()
        searchable = filtered[["payment_id", "order_id", "email", "contact"]].astype(str)
        filtered = filtered[
            searchable.apply(lambda column: column.str.casefold().str.contains(needle)).any(axis=1)
        ]
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]
    if methods:
        filtered = filtered[filtered["method"].isin(methods)]
    if currencies:
        filtered = filtered[filtered["currency"].isin(currencies)]
    if risk_statuses:
        filtered = filtered[filtered["risk_status"].isin(risk_statuses)]
    if international_only:
        filtered = filtered[filtered["international"]]
    if is_mock and geography_filter != "Any":
        filtered = filtered[filtered["ip_billing_mismatch"].eq(geography_filter == "Mismatch")]
    if is_mock and device_filter != "Any":
        filtered = filtered[filtered["new_device"].eq(device_filter == "New")]
    captured = filtered[filtered["status"].eq("captured")]
    captured_values = captured.groupby("currency")["amount"].sum().to_dict()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions", f"{len(filtered):,}")
    m2.metric("Captured", f"{len(captured):,}")
    m3.metric("Failed", f"{filtered['status'].eq('failed').sum():,}")
    m4.metric("Captured value", format_currency_costs(captured_values))
    loaded_range = st.session_state.get("razorpay_loaded_range")
    if loaded_range:
        st.caption(
            f"Showing {len(filtered):,} of {len(transactions):,} transactions · "
            f"{loaded_range[0]:%d %b %Y} – {loaded_range[1]:%d %b %Y} UTC"
        )
    if is_mock:
        displayed = pd.DataFrame({
            "Txn": filtered["payment_id"],
            "Amount": filtered.apply(lambda row: f"{row['currency']} {row['amount']:,.2f}", axis=1),
            "Method": filtered["method"].str.title(),
            "Velocity": filtered["velocity"].map(lambda value: f"{int(value)}/hr"),
            "IP/Billing": filtered["ip_billing_mismatch"].map({True: "Mismatch", False: "Match"}),
            "Device": filtered["new_device"].map({True: "New", False: "Known"}),
            "Score": filtered["risk_score"].astype(float),
            "Risk": filtered["risk_status"],
            "Payment status": filtered["status"].str.title(),
        })
    else:
        st.info(
            "Razorpay does not supply device history, IP comparison, velocity, or fraud labels, "
            "so real payments are not assigned a fabricated score."
        )
        displayed = filtered[[
            "payment_id", "created_at", "amount", "currency", "status", "method",
            "order_id", "email", "contact",
        ]]
    st.dataframe(
        displayed, use_container_width=True, hide_index=True, height=470,
        column_config={"Score": st.column_config.ProgressColumn(
            "Risk score", min_value=0.0, max_value=1.0, format="%.3f"
        )},
    )
    st.download_button(
        "Download filtered transactions", csv_injection_safe(displayed).to_csv(index=False),
        "razorpay_transactions.csv", mime="text/csv",
    )
    if is_mock and filtered["risk_score"].ge(DEMO_BLOCKING_THRESHOLD).any():
        st.caption(
            "High-risk rows here can be opened in **Transaction investigation** for the full "
            "signal breakdown, AI evidence report, and case actions."
        )


def render_global_importance() -> None:
    st.markdown("### What drives the model's score")
    st.caption("Global SHAP importance across the full held-out test set — not specific to one transaction.")
    try:
        importance = load_global_importance(GLOBAL_IMPORTANCE_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.info(f"Global feature importance unavailable: {exc}")
        return
    rows = "".join(
        f'<div class="importance-row"><span>{signal}</span>'
        f'<div class="importance-track"><div class="importance-fill" style="width:{percent}%"></div></div>'
        f'<strong>{percent:.1f}%</strong></div>'
        for signal, percent in importance["signal_importance_percent"].items()
    )
    st.markdown(f'<div class="importance-panel">{rows}</div>', unsafe_allow_html=True)


def render_model_transparency_section(transactions: pd.DataFrame) -> None:
    """The trained model's held-out performance, cost explorer, and global importance.

    Embedded in Fraud overview rather than its own nav item: this is real,
    static model evaluation content, not something that needs a page of its own.
    """
    st.divider()
    st.markdown("### Held-out model performance")
    evaluation_path = PROJECT_ROOT / "backend" / "reports" / "metrics" / "evaluation.json"
    try:
        evaluation = json.loads(evaluation_path.read_text())
        selected = evaluation["chosen"]
        ranking = evaluation["ranking"]
        curve = load_threshold_curve(THRESHOLD_CURVE_PATH)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        st.info(f"Evaluation artifacts unavailable; run `python -m backend.src.train`. ({exc})")
        render_global_importance()
        return
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Precision", f"{selected['precision']:.1%}")
    h2.metric("Recall", f"{selected['recall']:.1%}")
    h3.metric("F1", f"{selected['f1']:.1%}")
    h4.metric("False-positive rate", f"{selected['false_positive_rate']:.1%}")
    h5.metric("PR-AUC", f"{float(ranking['average_precision']):.3f}")
    st.caption(
        f"Newest 30% held-out test window at the validation-selected threshold "
        f"{selected['threshold']:.3f}. Illustrative error cost: "
        f"${selected['total_cost']:,.0f}. Synthetic data only."
    )
    st.info(
        "This is prototype validation on generated labels, not production accuracy. "
        "SHAP reasons identify model contributors, not proof that a payment is fraudulent."
    )
    st.markdown("### Threshold sensitivity explorer")
    st.caption(
        "Descriptive what-if analysis on the held-out test set. It does not replace the "
        "validation-selected threshold and must not be treated as a production policy."
    )
    left, right = st.columns(2)
    with left:
        cost_fp = st.slider("False-positive cost ($)", 1, 200, 5)
    with right:
        cost_fn = st.slider("False-negative cost ($)", 10, 2000, 500, 10)
    priced = cost_curve_for_ratio(curve, cost_fp, cost_fn)
    cheapest = cheapest_threshold_row(priced)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Lowest-cost test point", f"{cheapest['threshold']:.2f}")
    p2.metric("Precision there", f"{cheapest['precision']:.1%}")
    p3.metric("Recall there", f"{cheapest['recall']:.1%}")
    p4.metric("Descriptive cost", f"${cheapest['total_cost']:,.0f}")
    st.line_chart(priced.set_index("threshold")["total_cost"])
    render_global_importance()
    if transactions["risk_score"].notna().any():
        st.markdown("### Demonstration decision audit")
        st.caption("Synthetic labels evaluate the mock decisions; this is separate from real payments.")
        evidence = risk_evidence_summary(transactions, threshold=DEMO_BLOCKING_THRESHOLD)
        e1, e2, e3 = st.columns(3)
        e1.metric("Demo transactions", f"{evidence['transactions']:,}")
        e2.metric("Flagged for fraud", f"{evidence['blocked']:,}")
        e3.metric("Correctly caught", f"{evidence['correctly_caught']:,}")
        st.dataframe(
            risk_audit_rows(transactions, threshold=DEMO_BLOCKING_THRESHOLD),
            use_container_width=True, hide_index=True,
        )


def render_fraud_overview(transactions: pd.DataFrame) -> None:
    """KPIs, a risk trend, and the top open alerts across the loaded window."""
    render_page_header(
        "Fraud overview", "Portfolio-level risk posture across the loaded synthetic transactions."
    )
    if transactions.empty:
        st.info("No transactions are loaded for the selected date range.")
        return
    if transactions["risk_score"].notna().any():
        total = len(transactions)
        fraud = int(transactions["actual"].eq("Fraud").sum())
        flagged = int(transactions["risk_status"].eq("High risk").sum())
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Total transactions", f"{total:,}")
        o2.metric("Labelled fraud (synthetic)", f"{fraud:,}")
        o3.metric("Labelled legitimate", f"{total - fraud:,}")
        o4.metric("Flagged high risk", f"{flagged:,}")
        st.caption(
            "Fraud/legitimate labels are synthetic ground truth revealed only to evaluate the demo; "
            "they are not real Razorpay findings."
        )

        st.markdown("### Risk trend")
        trend = (
            transactions.sort_values("created_at")
            .set_index("created_at")["risk_score"]
            .rolling(5, min_periods=1).mean()
        )
        st.line_chart(trend, height=220)
        st.caption("5-transaction rolling average risk score over the loaded window.")

        st.markdown("### Top alerts")
        top_alerts = transactions.sort_values("risk_score", ascending=False).head(5)
        for _, row in top_alerts.iterrows():
            left, right = st.columns([5, 1.4], vertical_alignment="center")
            with left:
                st.markdown(
                    f"**{row['payment_id']}** · score **{row['risk_score']:.3f}** "
                    f"({row['risk_status']})  \n{' · '.join(_signal_reasons(row))}"
                )
            with right:
                st.button(
                    "Investigate →", key=f"overview_investigate_{row['payment_id']}",
                    use_container_width=True, on_click=_open_demo_view,
                    args=("Transaction investigation", str(row["payment_id"])),
                )
    else:
        st.info(
            "Fraud overview needs scored transactions. Real Razorpay payments have no FraudLens "
            "score because the Payments API does not supply the required signals."
        )
    render_model_transparency_section(transactions)
    st.markdown(
        '<div class="defense-banner">Defense-only: this view explains and prioritizes existing '
        'signals for human review. It never automatically captures or refunds a payment.</div>',
        unsafe_allow_html=True,
    )


def render_fraud_alerts(transactions: pd.DataFrame) -> None:
    """Prioritized review queue for every transaction above the review threshold."""
    render_page_header(
        "Fraud review queue",
        "Start with high-priority transactions, inspect the evidence, and record a human decision.",
    )
    if transactions.empty:
        st.info("No transactions are loaded for the selected date range.")
        return
    if not transactions["risk_score"].notna().any():
        st.info(
            "Alerts need scored transactions. Real Razorpay payments have no FraudLens score because "
            "the Payments API does not supply the required signals."
        )
        return
    alerts = transactions[transactions["risk_score"].ge(DEMO_REVIEW_THRESHOLD)].sort_values(
        ["risk_score", "created_at"], ascending=[False, False]
    )
    if alerts.empty:
        st.success("No transactions are at or above the review threshold in this window.")
        return
    high_priority = alerts["risk_score"].ge(DEMO_BLOCKING_THRESHOLD)
    high_count = int(high_priority.sum())
    review_count = len(alerts) - high_count
    st.markdown(
        f"""
        <div class="alert-queue-guide">
          <div>
            <strong>How to use this queue</strong>
            <span>Risk score ranks review urgency—it is not proof of fraud.</span>
          </div>
          <div class="alert-guide-bands">
            <span class="alert-band alert-band-high">High priority · {DEMO_BLOCKING_THRESHOLD:.0%}–100%</span>
            <span class="alert-band alert-band-review">Needs review · {DEMO_REVIEW_THRESHOLD:.0%}–{DEMO_BLOCKING_THRESHOLD - .01:.0%}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)
    a1.metric("Needs analyst review", f"{len(alerts):,}", help=f"Risk score at or above {DEMO_REVIEW_THRESHOLD:.0%}")
    a2.metric("High priority", f"{high_count:,}", help=f"Risk score at or above {DEMO_BLOCKING_THRESHOLD:.0%}")
    a3.metric("Review band", f"{review_count:,}", help=f"Risk score from {DEMO_REVIEW_THRESHOLD:.0%} to {DEMO_BLOCKING_THRESHOLD - .01:.0%}")

    filter_col, count_col = st.columns([3, 1], vertical_alignment="bottom")
    with filter_col:
        priority_filter = st.radio(
            "Show alerts",
            ["High priority", "All alerts", "Review band"],
            horizontal=True,
            key="fraud_alert_priority_filter",
        )
    if priority_filter == "High priority":
        visible_alerts = alerts.loc[high_priority]
    elif priority_filter == "Review band":
        visible_alerts = alerts.loc[~high_priority]
    else:
        visible_alerts = alerts
    with count_col:
        st.caption(f"Showing **{len(visible_alerts):,}** of **{len(alerts):,}** alerts")

    for _, row in visible_alerts.iterrows():
        score = float(row["risk_score"])
        is_high = score >= DEMO_BLOCKING_THRESHOLD
        priority_label = "HIGH PRIORITY" if is_high else "NEEDS REVIEW"
        priority_class = "high" if is_high else "review"
        next_step = (
            "Review the evidence before fulfillment and record the case outcome."
            if is_high else
            "Check customer context and escalate only when the evidence supports it."
        )
        reason_chips = "".join(
            f'<span class="alert-reason-chip">{escape(reason)}</span>'
            for reason in _signal_reasons(row)
        )
        created_at = pd.Timestamp(row["created_at"])
        with st.container(border=True):
            st.markdown(
                f'<span class="alert-card-anchor alert-card-{priority_class}"></span>',
                unsafe_allow_html=True,
            )
            details, action = st.columns([5, 1.45], vertical_alignment="center")
            with details:
                st.markdown(
                    f"""
                    <div class="alert-card-header">
                      <span class="alert-priority alert-priority-{priority_class}">{priority_label}</span>
                      <span class="alert-risk-score"><strong>{score:.0%}</strong> risk score</span>
                    </div>
                    <div class="alert-payment-line">
                      <strong>{escape(str(row['currency']))} {float(row['amount']):,.2f}</strong>
                      <span>{escape(str(row['payment_id']))}</span>
                    </div>
                    <div class="alert-payment-meta">
                      {escape(str(row['method']).title())} · {escape(str(row['status']).title())} ·
                      {created_at:%d %b %Y, %H:%M UTC}
                    </div>
                    <div class="alert-reason-label">WHY IT WAS FLAGGED</div>
                    <div class="alert-reason-list">{reason_chips}</div>
                    <div class="alert-next-step"><strong>Next step:</strong> {next_step}</div>
                    """,
                    unsafe_allow_html=True,
                )
            with action:
                st.button(
                    "Review transaction →",
                    key=f"alert_investigate_{row['payment_id']}",
                    type="primary" if is_high else "secondary",
                    use_container_width=True,
                    on_click=_open_demo_view,
                    args=("Transaction investigation", str(row["payment_id"])),
                )


def _render_transaction_chat(row: pd.Series) -> None:
    st.markdown("### Ask AI about this transaction")
    payment_id = str(row["payment_id"])
    chats = st.session_state.setdefault("razorpay_transaction_chats", {})
    history = chats.setdefault(payment_id, [])
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input(f"Ask about {payment_id}", key="investigation_question")
    if not question:
        return
    prior = history[-8:]
    history.append({"role": "user", "content": question})
    context = {
        "payment_id": payment_id, "transaction_id": payment_id,
        "timestamp": row["created_at"].isoformat(), "amount": float(row["amount"]),
        "currency": str(row["currency"]), "status": str(row["status"]),
        "method": str(row["method"]), "order_id": str(row["order_id"]),
        "email": str(row["email"]), "contact": str(row["contact"]),
        "international": bool(row["international"]),
        "velocity": None if pd.isna(row["velocity"]) else int(row["velocity"]),
        "ip_billing_mismatch": None if pd.isna(row["ip_billing_mismatch"]) else bool(row["ip_billing_mismatch"]),
        "new_device": None if pd.isna(row["new_device"]) else bool(row["new_device"]),
        "amount_deviation": None if pd.isna(row["amount_deviation"]) else float(row["amount_deviation"]),
        "risk_score": None if pd.isna(row["risk_score"]) else float(row["risk_score"]),
        "risk_status": None if pd.isna(row["risk_status"]) else str(row["risk_status"]),
        "actual": None if pd.isna(row["actual"]) else str(row["actual"]),
    }
    try:
        response = ask_preview_transaction_question(context, question, prior, SCORING_API_URL)
        answer = response["answer"] if response["status"] == "generated" else response.get("error", "Unable to answer.")
    except ScoringAPIError as exc:
        answer = str(exc)
    history.append({"role": "assistant", "content": answer})
    st.rerun()


def _render_case_actions(row: pd.Series) -> None:
    payment_id = str(row["payment_id"])
    st.markdown("### Case status")
    try:
        payload = get_fraud_case(payment_id, SCORING_API_URL)
        case, notes = payload.get("case"), payload.get("notes") or []
    except ScoringAPIError as exc:
        st.warning(f"Case storage unavailable: {exc}")
        case, notes = None, []
    current_status = case["status"] if case else "open"
    updated_by = f" · last updated by {case['updated_by']}" if case and case.get("updated_by") else ""
    st.caption(f"Current status: **{CASE_STATUS_LABELS.get(current_status, current_status)}**{updated_by}")

    analyst = st.text_input(
        "Analyst name", key="investigation_analyst_name", placeholder="Name or work email"
    )
    status_cols = st.columns(3)
    with status_cols[0]:
        mark_investigating = st.button(
            "Mark under investigation", use_container_width=True, key=f"case_investigating_{payment_id}"
        )
    with status_cols[1]:
        mark_fraud = st.button(
            "Confirm fraud", use_container_width=True, key=f"case_fraud_{payment_id}"
        )
    with status_cols[2]:
        mark_false_positive = st.button(
            "Mark false positive", use_container_width=True, key=f"case_fp_{payment_id}"
        )
    action = (
        "under_investigation" if mark_investigating
        else "confirmed_fraud" if mark_fraud
        else "false_positive" if mark_false_positive
        else None
    )
    if action:
        try:
            risk_score = None if pd.isna(row["risk_score"]) else float(row["risk_score"])
            set_fraud_case_status(payment_id, action, SCORING_API_URL, actor=analyst, risk_score=risk_score)
            st.success(f"Case marked {CASE_STATUS_LABELS[action]}.")
            st.rerun()
        except ScoringAPIError as exc:
            st.error(str(exc))

    note_text = st.text_area("Add a note", key=f"case_note_{payment_id}", placeholder="Investigation notes…")
    if st.button("Save note", key=f"case_save_note_{payment_id}"):
        if note_text.strip():
            try:
                add_fraud_case_note(payment_id, note_text, SCORING_API_URL, author=analyst)
                st.success("Note saved.")
                st.rerun()
            except ScoringAPIError as exc:
                st.error(str(exc))
        else:
            st.warning("Write a note before saving.")

    if notes:
        st.markdown("#### Notes")
        for note in notes:
            st.markdown(f"**{note.get('author') or 'Analyst'}** · {note.get('created_at', '')}  \n{note['note']}")


def render_transaction_investigation(transactions: pd.DataFrame, *, is_mock: bool) -> None:
    """Full signal detail, the AI evidence report, grounded chat, and case actions for one transaction."""
    render_page_header(
        "Transaction investigation",
        "Full signal detail, the AI evidence report, grounded chat, and case actions for one transaction.",
    )
    if transactions.empty:
        st.info("No transactions are loaded for the selected date range.")
        return
    options = transactions.sort_values(
        "risk_score", ascending=False, na_position="last"
    )["payment_id"].astype(str).tolist()
    if st.session_state.get("investigation_payment_id") not in options:
        st.session_state["investigation_payment_id"] = options[0]
    payment_id = st.selectbox("Transaction", options, key="investigation_payment_id")
    row = transactions.loc[transactions["payment_id"].astype(str).eq(payment_id)].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score", "n/a" if pd.isna(row["risk_score"]) else f"{float(row['risk_score']):.3f}")
    m2.metric("Risk status", "n/a" if pd.isna(row["risk_status"]) else str(row["risk_status"]))
    m3.metric("Payment status", str(row["status"]).title())
    st.caption(
        f"{row['currency']} {row['amount']:,.2f} · {row['method'].title()} · "
        f"{row['created_at']:%d %b %Y, %H:%M UTC}"
    )

    if pd.isna(row["risk_score"]):
        st.info(
            "This is a real Razorpay payment. The Payments API does not supply device history, IP "
            "geography, velocity, or fraud labels, so no FraudLens signals or score are shown."
        )
    else:
        st.markdown("### Fraud signals")
        st.dataframe(
            pd.DataFrame({
                "Signal": ["Transaction velocity", "IP/billing geography", "Device history", "Amount deviation"],
                "Observed value": [
                    f"{int(row['velocity'])} recent transactions",
                    "Mismatch" if row["ip_billing_mismatch"] else "Match",
                    "New device" if row["new_device"] else "Known device",
                    f"{float(row['amount_deviation']):+.0f}% vs. customer baseline",
                ],
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("### AI evidence report")
        if float(row["risk_score"]) < DEMO_BLOCKING_THRESHOLD:
            st.caption(
                "The AI evidence report is available once a transaction is at or above the "
                f"auto-block threshold ({DEMO_BLOCKING_THRESHOLD:.2f}); this one is below it."
            )
        else:
            if st.button("Generate full evidence report", type="primary", key="investigation_generate_report"):
                transaction = {
                    "payment_id": payment_id, "velocity": int(row["velocity"]),
                    "ip_billing_mismatch": bool(row["ip_billing_mismatch"]),
                    "new_device": bool(row["new_device"]),
                    "amount_deviation": float(row["amount_deviation"]),
                    "risk_score": float(row["risk_score"]),
                }
                try:
                    with st.spinner("Turning verified signals into a reviewer-ready explanation…"):
                        report = generate_demo_transaction_report(
                            transaction, SCORING_API_URL, threshold=DEMO_BLOCKING_THRESHOLD
                        )
                    st.session_state["demo_evidence_report"] = {"payment_id": payment_id, "report": report}
                except ScoringAPIError as exc:
                    st.error(str(exc))
            saved = st.session_state.get("demo_evidence_report")
            if saved and saved["payment_id"] == payment_id:
                report = saved["report"]
                if report.get("summary"):
                    st.info(report["summary"])
                elif report.get("error"):
                    st.warning(report["error"])
                st.dataframe(
                    pd.DataFrame({
                        "Signal": [item["signal"].replace("_", " ").title() for item in report["evidence"]],
                        "Observed evidence": [item["detail"] for item in report["evidence"]],
                    }),
                    use_container_width=True, hide_index=True,
                )
                st.caption(report["confidence_note"])

    if is_mock:
        _render_case_actions(row)
    else:
        st.caption("Case management is available in the synthetic demo.")

    _render_transaction_chat(row)


def render_case_management() -> None:
    """A filterable log of every transaction an analyst has marked, linking back to Investigation."""
    render_page_header(
        "Case management", "Every transaction an analyst has marked, filterable by status."
    )
    status_label = st.selectbox(
        "Status", ["All"] + list(CASE_STATUS_LABELS.values()), key="case_management_status_filter"
    )
    status_value = None
    if status_label != "All":
        status_value = next(key for key, label in CASE_STATUS_LABELS.items() if label == status_label)
    try:
        cases = list_fraud_cases(SCORING_API_URL, status=status_value)
    except ScoringAPIError as exc:
        st.error(str(exc))
        return
    if not cases:
        st.info("No cases have been opened yet. Mark a transaction from Transaction investigation to start one.")
        return
    table = pd.DataFrame([
        {
            "Transaction": case["transaction_id"],
            "Status": CASE_STATUS_LABELS.get(case["status"], case["status"]),
            "Risk score": case.get("risk_score"),
            "Updated by": case.get("updated_by") or "—",
            "Updated at": case.get("updated_at"),
        }
        for case in cases
    ])
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={"Risk score": st.column_config.NumberColumn("Risk score", format="%.3f")},
    )
    jump_left, jump_right = st.columns([3, 1], vertical_alignment="bottom")
    with jump_left:
        jump_to = st.selectbox(
            "Open a case in Transaction investigation", table["Transaction"].tolist(), key="case_management_jump"
        )
    with jump_right:
        st.button(
            "Open →", type="primary", use_container_width=True,
            on_click=_open_demo_view, args=("Transaction investigation", jump_to),
        )


def render_account_view(connection: dict) -> None:
    """Explain the connected-account boundary and offer a real OAuth switch."""
    render_page_header(
        "Razorpay account",
        "Connect Test Mode for payment-history review; model scoring remains synthetic until enrichment is available.",
    )
    if connection.get("mock"):
        st.info(
            "You are viewing the synthetic demo account. Real Razorpay payments are not scored by "
            "the trained model because the Payments API does not include device history, IP geography, "
            "velocity, and customer-history signals."
        )
        if RAZORPAY_MODE == "test" and oauth_is_configured():
            st.link_button(
                "Connect a real Razorpay Test Mode account",
                build_authorization_url(
                    RAZORPAY_CLIENT_ID, RAZORPAY_REDIRECT_URI, issue_oauth_state()
                ),
                type="primary",
                use_container_width=True,
            )
        else:
            st.caption(
                "Add the three Razorpay Partner OAuth values to .env to enable the optional real-account path."
            )
        return

    account_id = connection.get("razorpay_account_id", "")
    st.success(f"Connected to Razorpay Test Mode account `{account_id}`.")
    st.info(
        "This view is an account browser for payment history, filtering, export, and grounded chat. "
        "It is not a trained-model risk view: real Payments API rows are shown without a risk score "
        "or synthetic error label."
    )


if "razorpay_connection" not in st.session_state:
    if RAZORPAY_MOCK_AUTH:
        st.session_state["razorpay_connection"] = {
            "access_token": "mock-access-token",
            "razorpay_account_id": "acc_mock_demo",
            "mode": "test",
            "mock": True,
        }
    else:
        render_connection_screen()

connection = st.session_state["razorpay_connection"]
is_mock_session = bool(connection.get("mock"))
if not is_mock_session and connection.get("razorpay_account_id"):
    # The webhook process records an account.app.authorization_revoked event in the
    # shared review store. Clear this process's token on its next run as well.
    from backend.src.review_store import SupabaseStoreError, review_store_from_environment

    try:
        revoked = review_store_from_environment(RAZORPAY_ENFORCEMENT_DB).is_authorization_revoked(
            str(connection["razorpay_account_id"])
        )
    except SupabaseStoreError as exc:
        st.error(f"Review storage is unavailable: {exc}")
        st.stop()
    if revoked:
        for key in (
            "razorpay_connection", "razorpay_payments", "razorpay_loaded_range",
            "razorpay_transaction_chats", "fraudlens_view",
        ):
            st.session_state.pop(key, None)
        st.session_state["razorpay_revoked_notice"] = True
        st.rerun()
with st.sidebar:
    st.markdown(
        brand_lockup(text_size="1.2rem", text_color="#16233D", stroke="#5D94FC"),
        unsafe_allow_html=True,
    )
    # Streamlit renders `:material/name:` shortcodes in widget labels as real
    # Material Symbols vector icons (see streamlit.material_icon_names for
    # the valid name list), not emoji glyphs.
    navigation_icons = {
        "Review queue": "shield",
        "Fraud overview": "monitoring",
        "Fraud alerts": "notifications_active",
        "Transaction explorer": "travel_explore",
        "Transaction investigation": "manage_search",
        "Case management": "folder_open",
        "Razorpay account": "account_balance_wallet",
    }
    navigation = list(navigation_icons)
    if st.session_state.get("fraudlens_view") not in navigation:
        st.session_state["fraudlens_view"] = navigation[0]
    view = st.radio(
        "Navigation",
        navigation,
        index=0,
        key="fraudlens_view",
        format_func=lambda item: f":material/{navigation_icons[item]}:  {item}",
    )
    st.markdown(
        '<div class="defense-footer">Defense-only. No model decision automatically changes a payment.</div>',
        unsafe_allow_html=True,
    )

top_left, top_right = st.columns([4, 1], vertical_alignment="center")
account_id = connection.get("razorpay_account_id", "")
connection_name = "Demo mode" if is_mock_session else "Razorpay connected"
with top_left:
    connection_detail = (
        "Synthetic UI fixtures · rule-based demo scores"
        if is_mock_session
        else account_id
    )
    st.markdown(
        f'<div class="connection-bar"><span class="connection-led"></span>{connection_name} · '
        f'{connection_detail}</div>',
        unsafe_allow_html=True,
    )
disconnect = False
if not is_mock_session:
    with top_right:
        disconnect = st.button("Disconnect", use_container_width=True)
if disconnect:
    if not is_mock_session:
        try:
            revoke_access_token(
                connection["access_token"], client_id=RAZORPAY_CLIENT_ID,
                client_secret=RAZORPAY_CLIENT_SECRET,
            )
        except RazorpayOAuthError as exc:
            st.error(f"Account is still connected: {exc}")
            st.stop()
    for key in (
        "razorpay_connection", "razorpay_payments", "razorpay_loaded_range",
        "razorpay_transaction_chats", "fraudlens_view",
    ):
        st.session_state.pop(key, None)
    st.rerun()

if view == "Review queue":
    if is_mock_session:
        demo_transactions = load_dashboard_transactions(show_date_filter=False)
        render_mock_demo_guide(demo_transactions)
        render_mock_enforcement_panel()
    else:
        render_page_header("Review queue", "Human-approved Test Mode capture and refund actions.")
        render_enforcement_panel(connection)
elif view == "Razorpay account":
    render_account_view(connection)
elif view == "Case management":
    render_case_management()
else:
    transactions = load_dashboard_transactions(show_date_filter=view == "Transaction explorer")
    if transactions.empty:
        st.info("No transactions were found for the selected date range.")
    elif view == "Fraud overview":
        render_fraud_overview(transactions)
    elif view == "Fraud alerts":
        render_fraud_alerts(transactions)
    elif view == "Transaction explorer":
        render_transactions_view(transactions, is_mock=is_mock_session)
    elif view == "Transaction investigation":
        render_transaction_investigation(transactions, is_mock=is_mock_session)
