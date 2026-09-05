"""FraudLens transaction review dashboard."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import sys
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    normalize_uploaded_transactions,
    risk_audit_rows,
    risk_evidence_summary,
    score_and_save_uploaded_dataset,
    set_fraud_case_status,
    uploaded_scores_to_dashboard_transactions,
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

FRAUDGUARD_SHIELD_SVG = (
    '<svg viewBox="0 0 48 48" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M24 3.5 41 10v13.1c0 10.4-6.8 18.3-17 21.4C13.8 41.4 7 33.5 7 23.1V10L24 3.5Z" '
    'fill="#4f7df3"/>'
    '<path d="M24 9.5v28.2c6.7-2.6 10.9-8.1 10.9-14.6v-8.7L24 9.5Z" fill="#2f5fd7"/>'
    '<path d="M24 12.4 14.1 16.1v7.1c0 5.8 3.8 10.7 9.9 13.2V12.4Z" fill="#7ea5ff"/>'
    '<path d="M24 12.4v24c6.1-2.5 9.9-7.4 9.9-13.2v-7.1L24 12.4Z" fill="#5f8cff"/>'
    '</svg>'
)


def brand_lockup(size: str = "26px", text_size: str = "1.05rem", text_color: str = "#eef4ff", stroke: str = "#5d94fc") -> str:
    mark = CUT_LENS_MARK_SVG.format(stroke=stroke)
    return (
        f'<div class="brand-lockup" style="font-size:{text_size};color:{text_color};">'
        f'<span class="brand-mark" style="width:{size};height:{size};">{mark}</span>'
        f'<span>Fraud<span style="color:{stroke};">Lens</span></span></div>'
    )


def fraudguard_brand_lockup() -> str:
    return (
        '<div class="fraudguard-brand-lockup">'
        f'<span class="fraudguard-brand-mark">{FRAUDGUARD_SHIELD_SVG}</span>'
        '<div class="fraudguard-brand-text">'
        '<span class="fraudguard-brand-name">FraudLens</span>'
        '<span class="fraudguard-brand-tagline">Detect <b>•</b> Analyze <b>•</b> Prevent</span>'
        '</div>'
        '</div>'
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

MODEL_LABELS = {
    "tuned_xgboost_uncalibrated": "XGBoost (tuned)",
    "tuned_xgboost_calibrated": "XGBoost (tuned, calibrated)",
    "random_forest": "Random forest",
    "logistic_regression": "Logistic regression",
}

RISK_STATUS_PILL_LABELS = {"High risk": "Flagged", "Review": "Review", "Low risk": "Legit"}
PILL_COLORS = {
    "Flagged": ("#d13b3b", "rgba(209, 59, 59, 0.09)"),
    "Review": ("#b3760f", "rgba(179, 118, 15, 0.1)"),
    "Legit": ("#1a7f37", "rgba(26, 127, 55, 0.1)"),
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


def _risk_indicator_cards(row: pd.Series) -> list[dict]:
    """Icon-card view of the same four real signals as the fraud-signals table."""
    velocity = int(row["velocity"])
    deviation = float(row["amount_deviation"])
    mismatch = bool(row["ip_billing_mismatch"])
    new_device = bool(row["new_device"])
    return [
        {
            "icon": "trending_up",
            "title": "Transaction velocity",
            "detail": f"{velocity} recent transactions",
            "severity": "high" if velocity >= 8 else "medium" if velocity >= 4 else "low",
        },
        {
            "icon": "location_on",
            "title": "IP/billing geography",
            "detail": "Mismatch" if mismatch else "Match",
            "severity": "high" if mismatch else "low",
        },
        {
            "icon": "devices",
            "title": "Device history",
            "detail": "New device" if new_device else "Known device",
            "severity": "high" if new_device else "low",
        },
        {
            "icon": "payments",
            "title": "Amount deviation",
            "detail": f"{deviation:+.0f}% vs. baseline",
            "severity": "high" if abs(deviation) >= 75 else "medium" if abs(deviation) >= 30 else "low",
        },
    ]


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
          <h2>Payment review</h2>
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


def _sync_csv_upload_state() -> None:
    """Discard scored results as soon as the selected CSV changes."""
    if "csv_tester_upload" not in st.session_state:
        return
    uploaded = st.session_state.get("csv_tester_upload")
    current_file_id = getattr(uploaded, "file_id", None)
    cached = st.session_state.get("csv_tester_results")
    if cached and cached.get("file_id") != current_file_id:
        st.session_state.pop("csv_tester_results", None)
        st.session_state.pop("csv_tester_active", None)


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
    shortcut_columns = st.columns(3)
    with shortcut_columns[0]:
        st.button(
            "Investigate top alert",
            key="demo_open_investigation",
            type="primary",
            use_container_width=True,
            disabled=report_payment_id is None,
            on_click=_open_demo_view,
            args=("Transaction investigation", report_payment_id),
        )
    with shortcut_columns[1]:
        st.button(
            "Explore all transactions",
            key="demo_open_explorer",
            use_container_width=True,
            on_click=_open_demo_view,
            args=("Transaction explorer",),
        )
    with shortcut_columns[2]:
        st.button(
            "Open case management",
            key="demo_open_cases",
            use_container_width=True,
            on_click=_open_demo_view,
            args=("Case management",),
        )
    st.caption(
        "The action panel below this tour covers Authorized → Approve & capture, Authorized → Confirm "
        "fraud & release authorization, and Captured → Refund & stop fulfillment. Every action is "
        "session-only in the demo session and is recorded in the visible audit log."
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


def _render_fraud_parameters_panel(
    filtered: pd.DataFrame,
    *,
    signal_importance: dict[str, float] | None = None,
    signal_support: dict[str, float] | None = None,
    upload_active: bool = False,
    upload_pending: bool = False,
    upload_filename: str | None = None,
    upload_row_count: int | None = None,
    upload_flagged_count: int | None = None,
    decision_threshold: float | None = None,
) -> None:
    """Full-width panel: measured SHAP signal influence for the active dataset."""
    with st.container(border=True):
        st.markdown('<span class="explorer-panel-anchor"></span>', unsafe_allow_html=True)
        title = "Model signal influence" if upload_active else "Fraud detection parameters"
        st.markdown(
            f'<div class="explorer-panel-title">:material/tune: {title}</div>',
            unsafe_allow_html=True,
        )
        if upload_pending:
            st.info("Run the model to calculate signal influence for this CSV.")
            return
        if upload_active and not isinstance(signal_importance, dict):
            st.info("Dataset signal influence is unavailable for this scored CSV.")
            return
        if upload_active and not isinstance(signal_support, dict):
            st.info("Dataset signal coverage is unavailable for this scored CSV.")
            return
        caption = (
            "Mean absolute SHAP influence from the trained XGBoost model on this uploaded CSV."
            if upload_active
            else "Real mean absolute SHAP signal importance from the held-out test set — read-only."
        )
        st.markdown(f'<div class="explorer-panel-caption">{caption}</div>', unsafe_allow_html=True)
        if upload_active:
            row_count = int(upload_row_count) if upload_row_count is not None else len(filtered)
            flagged_count = (
                int(upload_flagged_count) if upload_flagged_count is not None else None
            )
            metadata = [
                f"Dataset: {str(upload_filename or 'uploaded CSV')}",
                f"{row_count:,} rows scored",
            ]
            if flagged_count is not None:
                metadata.append(f"{flagged_count:,} flagged")
            if decision_threshold is not None:
                metadata.append(f"review threshold {float(decision_threshold):.3f}")
            st.caption(" · ".join(metadata))
        try:
            importance_values = (
                signal_importance
                if upload_active
                else load_global_importance(GLOBAL_IMPORTANCE_PATH)["signal_importance_percent"]
            )
            if not importance_values:
                st.info("Dataset signal influence is unavailable for this scored CSV.")
                return
            if upload_active:
                top_signal, top_percent = max(
                    importance_values.items(), key=lambda item: float(item[1])
                )
                st.markdown(
                    f"**Top model driver:** {escape(str(top_signal))} · "
                    f"{float(top_percent):.1f}% of total mean absolute SHAP influence"
                )
            def importance_row(signal: str, percent: float) -> str:
                coverage = (
                    f'<small>Contributed in {float(signal_support.get(signal, 0.0)):.1f}% of scored rows</small>'
                    if upload_active else ""
                )
                return (
                    f'<div class="importance-row"><div class="importance-signal">'
                    f'<span>{escape(str(signal))}</span>{coverage}</div>'
                    f'<div class="importance-track"><div class="importance-fill" style="width:{percent}%"></div></div>'
                    f'<strong>{percent:.1f}%</strong></div>'
                )

            rows = "".join(
                importance_row(str(signal), float(percent))
                for signal, percent in importance_values.items()
            )
            st.markdown(f'<div class="importance-panel">{rows}</div>', unsafe_allow_html=True)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            st.info(f"Signal importance unavailable: {exc}")


def render_transactions_view(transactions: pd.DataFrame, *, is_mock: bool) -> None:
    st.markdown('<span class="transaction-explorer-route-label">Transaction explorer</span>', unsafe_allow_html=True)
    _sync_csv_upload_state()
    upload_result = st.session_state.get("csv_tester_results")
    using_upload = bool(upload_result and st.session_state.get("csv_tester_active"))
    upload_pending = st.session_state.get("csv_tester_upload") is not None and not using_upload
    if using_upload:
        transactions = uploaded_scores_to_dashboard_transactions(
            upload_result["scored"],
            review_threshold=float(upload_result.get("decision_threshold") or DEMO_REVIEW_THRESHOLD),
        )
    has_model_scores = is_mock or using_upload
    loaded_range = st.session_state.get("razorpay_loaded_range")
    header_col, meta_col = st.columns([2.2, 1.3], vertical_alignment="top")
    with header_col:
        render_page_header(
            "Uploaded transaction results" if using_upload else "Transaction explorer" if is_mock else "Account transactions",
            f"Reviewing {len(transactions):,} transactions scored from the uploaded CSV."
            if using_upload else "Explore transactions, apply filters, and understand how fraud risk is computed."
            if is_mock else "Payment history loaded from the connected Razorpay Test Mode account.",
        )
    with meta_col:
        if loaded_range and not using_upload:
            st.markdown(
                '<div class="connection-bar" style="justify-content:center;margin-top:0.4rem;">'
                f'<span class="connection-led"></span>{loaded_range[0]:%d %b %Y} – {loaded_range[1]:%d %b %Y}</div>',
                unsafe_allow_html=True,
            )
        st.download_button(
            "Export", csv_injection_safe(transactions).to_csv(index=False),
            "razorpay_transactions.csv", mime="text/csv", use_container_width=True,
            key="explorer_export_all",
        )
    if is_mock and not using_upload:
        st.info("Demo signals are simulated; they are not real Razorpay fraud findings.")
    elif using_upload:
        st.info("The table, totals, flagged count, averages, and signal influence below use the uploaded CSV.")

    prefill = st.session_state.pop("explorer_search_prefill", None)
    if prefill is not None:
        st.session_state["explorer_filter_search"] = prefill

    with st.container(border=True):
        st.markdown('<span class="filters-card-anchor"></span>', unsafe_allow_html=True)
        title_col, clear_col = st.columns([4, 1], vertical_alignment="center")
        with title_col:
            st.markdown(
                '<div class="filters-card-title">:material/filter_alt: Filters</div>', unsafe_allow_html=True
            )
        with clear_col:
            clear_clicked = st.button("Clear all", key="explorer_clear_filters", use_container_width=True)
        if clear_clicked:
            for key in (
                "explorer_filter_search", "explorer_filter_status", "explorer_filter_method",
                "explorer_filter_currency", "explorer_filter_risk_status", "explorer_filter_international",
                "explorer_geography_filter", "explorer_device_filter", "explorer_amount_min",
                "explorer_amount_max", "explorer_risk_min", "explorer_risk_max",
            ):
                st.session_state.pop(key, None)
            st.rerun()
        one, two, three, four = st.columns(4)
        with one:
            search = st.text_input(
                "Search", placeholder="Payment ID, order ID, email…", key="explorer_filter_search"
            )
            statuses = st.multiselect(
                "Status", sorted(value for value in transactions["status"].dropna().unique() if value),
                key="explorer_filter_status",
            )
        with two:
            methods = st.multiselect(
                "Payment method", sorted(value for value in transactions["method"].dropna().unique() if value),
                key="explorer_filter_method",
            )
            currencies = st.multiselect(
                "Currency", sorted(value for value in transactions["currency"].dropna().unique() if value),
                key="explorer_filter_currency",
            )
        with three:
            st.caption("Amount range")
            amount_min_col, amount_max_col = st.columns(2)
            with amount_min_col:
                min_amount = st.number_input(
                    "Amount min", value=None, placeholder="Min", label_visibility="collapsed",
                    key="explorer_amount_min",
                )
            with amount_max_col:
                max_amount = st.number_input(
                    "Amount max", value=None, placeholder="Max", label_visibility="collapsed",
                    key="explorer_amount_max",
                )
            international_only = st.checkbox(
                "International payments only", key="explorer_filter_international"
            )
        with four:
            risk_statuses = []
            min_risk = max_risk = None
            if has_model_scores and transactions["risk_status"].notna().any():
                st.caption("Risk score range (0–1)")
                risk_min_col, risk_max_col = st.columns(2)
                with risk_min_col:
                    min_risk = st.number_input(
                        "Risk min", value=None, placeholder="Min", label_visibility="collapsed",
                        min_value=0.0, max_value=1.0, step=0.05, key="explorer_risk_min",
                    )
                with risk_max_col:
                    max_risk = st.number_input(
                        "Risk max", value=None, placeholder="Max", label_visibility="collapsed",
                        min_value=0.0, max_value=1.0, step=0.05, key="explorer_risk_max",
                    )
                risk_statuses = st.multiselect(
                    "Risk status",
                    [v for v in ("High risk", "Review", "Low risk") if v in set(transactions["risk_status"])],
                    key="explorer_filter_risk_status",
                )
        if has_model_scores:
            geo_col, dev_col = st.columns(2)
            with geo_col:
                geography_filter = st.selectbox(
                    "Geography", ["Any", "Match", "Mismatch"], key="explorer_geography_filter"
                )
            with dev_col:
                device_filter = st.selectbox(
                    "Device", ["Any", "Known", "New"], key="explorer_device_filter"
                )
        else:
            geography_filter = device_filter = "Any"
        st.write("")

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
    if min_amount is not None:
        filtered = filtered[filtered["amount"] >= min_amount]
    if max_amount is not None:
        filtered = filtered[filtered["amount"] <= max_amount]
    if has_model_scores and min_risk is not None:
        filtered = filtered[filtered["risk_score"] >= min_risk]
    if has_model_scores and max_risk is not None:
        filtered = filtered[filtered["risk_score"] <= max_risk]
    if risk_statuses:
        filtered = filtered[filtered["risk_status"].isin(risk_statuses)]
    if international_only:
        filtered = filtered[filtered["international"]]
    if has_model_scores and geography_filter != "Any":
        filtered = filtered[filtered["ip_billing_mismatch"].eq(geography_filter == "Mismatch")]
    if has_model_scores and device_filter != "Any":
        filtered = filtered[filtered["new_device"].eq(device_filter == "New")]

    _render_csv_batch_tester()

    if has_model_scores:
        _render_fraud_parameters_panel(
            filtered,
            signal_importance=(upload_result or {}).get("signal_importance_percent")
            if using_upload else None,
            signal_support=(upload_result or {}).get("signal_support_percent")
            if using_upload else None,
            upload_active=using_upload,
            upload_pending=upload_pending,
            upload_filename=(upload_result or {}).get("filename") if using_upload else None,
            upload_row_count=(upload_result or {}).get("row_count") if using_upload else None,
            upload_flagged_count=(
                int(upload_result["scored"]["flagged"].astype(bool).sum())
                if using_upload else None
            ),
            decision_threshold=(upload_result or {}).get("decision_threshold")
            if using_upload else None,
        )
    else:
        st.caption("Fraud detection parameters are available in the synthetic demo.")

    total_values = filtered.groupby("currency")["amount"].sum().to_dict()
    avg_currency = filtered["currency"].mode().iat[0] if filtered["currency"].notna().any() else ""
    avg_amount = filtered["amount"].mean() if len(filtered) else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(":material/database: Total transactions", f"{len(filtered):,}")
    if has_model_scores:
        flagged = int(filtered["risk_status"].eq("High risk").sum())
        flagged_pct = (flagged / len(filtered) * 100) if len(filtered) else 0.0
        m2.metric(
            ":material/warning: Flagged transactions", f"{flagged:,}",
            delta=f"{flagged_pct:.1f}%", delta_color="inverse",
        )
    else:
        m2.metric(":material/check_circle: Captured", f"{filtered['status'].eq('captured').sum():,}")
    m3.metric(":material/payments: Total transaction amount", format_currency_costs(total_values))
    m4.metric(
        ":material/bar_chart: Avg. transaction amount",
        format_currency_costs({avg_currency: avg_amount}) if avg_currency else "₹0.00",
    )
    st.caption(f"Showing {len(filtered):,} of {len(transactions):,} loaded transactions.")

    st.markdown(f"#### Transactions ({len(filtered):,})")
    if has_model_scores:
        if using_upload:
            displayed = pd.DataFrame({
                "Transaction ID": filtered["payment_id"],
                "Date & Time": filtered["created_at"].dt.strftime("%d %b %Y, %I:%M %p"),
                "User": filtered["email"].where(filtered["email"].ne(""), filtered["contact"]),
                "Amount": filtered.apply(lambda row: f"{row['currency']} {row['amount']:,.2f}", axis=1),
                "Method": filtered["method"].str.title(),
                "Velocity/hr": pd.to_numeric(filtered["velocity"], errors="coerce").round(2),
                "Geo mismatch": filtered["ip_billing_mismatch"].map({1.0: "Mismatch", 0.0: "Match", True: "Mismatch", False: "Match"}).fillna("Unknown"),
                "Risk score": (filtered["risk_score"] * 100).round().astype("Int64"),
                "Status": filtered["risk_status"].map(RISK_STATUS_PILL_LABELS),
                "Reasons": filtered["reasons"].map(
                    lambda items: "; ".join(items) if isinstance(items, list) else str(items)
                ),
            })
        else:
            displayed = pd.DataFrame({
                "Transaction ID": filtered["payment_id"],
                "Date & Time": filtered["created_at"].dt.strftime("%d %b %Y, %I:%M %p"),
                "Customer": filtered["email"].where(filtered["email"].ne(""), filtered["contact"]),
                "Amount": filtered.apply(lambda row: f"{row['currency']} {row['amount']:,.2f}", axis=1),
                "Payment method": filtered["method"].str.title(),
                "Risk score": (filtered["risk_score"] * 100).round().astype("Int64"),
                "Status": filtered["risk_status"].map(RISK_STATUS_PILL_LABELS),
                "Payment status": filtered["status"].str.title(),
            })

        def _style_row(row: pd.Series) -> list[str]:
            color, bg = PILL_COLORS.get(row["Status"], ("#5c6472", "#f0f2f5"))
            styles = []
            for column in row.index:
                if column == "Risk score":
                    styles.append(f"color:{color}; font-weight:600;")
                elif column == "Status":
                    styles.append(
                        f"color:{color}; background-color:{bg}; border-radius:999px; "
                        "font-weight:600; text-align:center;"
                    )
                else:
                    styles.append("")
            return styles

        styled = displayed.style.apply(_style_row, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True, height=470)
    else:
        st.info(
            "Razorpay does not supply device history, IP comparison, velocity, or fraud labels, "
            "so real payments are not assigned a fabricated score."
        )
        displayed = filtered[[
            "payment_id", "created_at", "amount", "currency", "status", "method",
            "order_id", "email", "contact",
        ]]
        st.dataframe(displayed, use_container_width=True, hide_index=True, height=470)

    st.download_button(
        "Download filtered transactions", csv_injection_safe(displayed).to_csv(index=False),
        "razorpay_transactions.csv", mime="text/csv", key="explorer_download_filtered",
    )
    if is_mock and not using_upload and len(filtered):
        jump_col, action_col = st.columns([3, 1], vertical_alignment="bottom")
        with jump_col:
            jump_to = st.selectbox(
                "Open a transaction in Transaction investigation",
                filtered["payment_id"].tolist(), key="explorer_jump_to",
            )
        with action_col:
            st.button(
                "Open →", type="primary", use_container_width=True, key="explorer_jump_button",
                on_click=_open_demo_view, args=("Transaction investigation", jump_to),
            )
    elif using_upload and len(filtered):
        jump_col, action_col = st.columns([3, 1], vertical_alignment="bottom")
        with jump_col:
            jump_to = st.selectbox(
                "Open an uploaded transaction in Transaction investigation",
                filtered["payment_id"].tolist(), key="explorer_upload_jump_to",
            )
        with action_col:
            st.button(
                "Open →", type="primary", use_container_width=True, key="explorer_upload_jump_button",
                on_click=_open_demo_view, args=("Transaction investigation", jump_to),
            )


def active_transaction_dataset(transactions: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Return the uploaded scored dataset when it is the active review source."""
    _sync_csv_upload_state()
    upload_result = st.session_state.get("csv_tester_results")
    if upload_result and st.session_state.get("csv_tester_active"):
        return uploaded_scores_to_dashboard_transactions(
            upload_result["scored"],
            review_threshold=float(upload_result.get("decision_threshold") or DEMO_REVIEW_THRESHOLD),
        ), True
    return transactions, False


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

    Embedded in Overview rather than its own nav item: this is real,
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


def _render_overview_steps(transactions: pd.DataFrame) -> None:
    """The Input → Detect → Results → Action pipeline, backed by the loaded window's real counts."""
    total = len(transactions)
    flagged = int(transactions["risk_status"].eq("High risk").sum())
    st.markdown(
        f"""
        <div class="workflow workflow-four">
          <div class="workflow-step">
            <div class="workflow-number">STEP 1</div>
            <div class="workflow-title">Input data</div>
            <p>{total:,} transactions loaded for the selected window.</p>
          </div>
          <div class="workflow-step">
            <div class="workflow-number">STEP 2</div>
            <div class="workflow-title">Detect spikes</div>
            <p>FraudLens scores every transaction with statistical and ML signals.</p>
          </div>
          <div class="workflow-step">
            <div class="workflow-number">STEP 3</div>
            <div class="workflow-title">View results</div>
            <p>Fraud trend, spike alerts, and the high-risk queue, below.</p>
          </div>
          <div class="workflow-step">
            <div class="workflow-number">STEP 4</div>
            <div class="workflow-title">Take action</div>
            <p>{flagged:,} transactions flagged high risk—investigate or decide now.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _overview_status_badge(score: float) -> tuple[str, str]:
    if score >= DEMO_BLOCKING_THRESHOLD:
        return "Flagged", "flagged"
    if score >= DEMO_REVIEW_THRESHOLD:
        return "Review", "review"
    return "Clear", "clear"


def _render_overview_infographic(transactions: pd.DataFrame, alerts: pd.DataFrame) -> None:
    """A reference-style overview panel: process cards, spike trend, high-risk table, and callout."""
    total = len(transactions)
    flagged = int(transactions["risk_status"].eq("High risk").sum())
    dated = transactions.assign(day=pd.to_datetime(transactions["created_at"], utc=True).dt.floor("D"))
    daily = dated.groupby("day").agg(
        total=("payment_id", "count"),
        flagged=("risk_status", lambda status: status.eq("High risk").sum()),
    )
    if daily.empty:
        daily = pd.DataFrame({"total": [0], "flagged": [0]}, index=[pd.Timestamp.utcnow().floor("D")])
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    if len(full_index) > 7:
        full_index = full_index[-7:]
    daily = daily.reindex(full_index, fill_value=0)
    total_points = daily["total"].astype(float).tolist()
    flagged_points = daily["flagged"].astype(float).tolist()
    labels = [pd.Timestamp(day).strftime("%b %d") for day in daily.index]

    def points(values: list[float], *, height: int = 210, width: int = 430) -> str:
        if not values:
            return ""
        max_value = max(max(total_points or [1]), max(flagged_points or [1]), 1)
        step = width / max(len(values) - 1, 1)
        coords = []
        for index, value in enumerate(values):
            x = 16 + index * step
            y = 18 + (height - 36) * (1 - (value / max_value))
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    latest_day = pd.Timestamp(daily.index[-1]).strftime("%b %d, %Y")
    previous_flagged = daily["flagged"].iloc[:-1]
    previous_average = float(previous_flagged.mean()) if len(previous_flagged) else 0.0
    latest_flagged = float(daily["flagged"].iloc[-1])
    if previous_average > 0:
        increase = int(round(((latest_flagged - previous_average) / previous_average) * 100))
    else:
        increase = 100 if latest_flagged > 0 else 0
    increase_label = f"{increase:+d}%"
    spike_title = "Fraud Spike Detected!" if increase > 0 else "Fraud Trend Stable"
    spike_copy = (
        "There is a significant increase in fraudulent transactions compared to the usual pattern."
        if increase > 0
        else "The latest uploaded transactions do not show an increase over the previous daily pattern."
    )
    spike_label_x = 352
    spike_label_y = 42

    table_rows = []
    high_risk_rows = alerts.sort_values(
        ["risk_score", "created_at"], ascending=[False, False], na_position="last"
    ).head(5)
    for _, row in high_risk_rows.iterrows():
        txn_id = str(row["payment_id"])
        created_at = pd.Timestamp(row["created_at"]).strftime("%b %d, %H:%M")
        amount_label = f"{float(row['amount']):,.0f}"
        score = int(round(float(row["risk_score"]) * 100))
        status_label = "Flagged" if str(row.get("risk_status")) == "High risk" else "Review"
        status_class = "flagged" if status_label == "Flagged" else "review"
        table_rows.append(
            f"""
            <tr>
              <td>{escape(txn_id)}</td>
              <td>{escape(created_at)}</td>
              <td>{escape(amount_label)}</td>
              <td>{score}</td>
              <td><span class="overview-status overview-status-{status_class}">{status_label}</span></td>
            </tr>
            """
        )
    rows_html = "".join(table_rows) or (
        '<tr><td colspan="5">No uploaded transactions are above the review threshold.</td></tr>'
    )
    x_labels = "".join(
        f'<span style="left:{(index / max(len(labels) - 1, 1)) * 100:.1f}%">{escape(label)}</span>'
        for index, label in enumerate(labels)
    )

    overview_css = (PROJECT_ROOT / "frontend" / "redesign.css").read_text()
    components.html(
        dedent(f"""
        <style>
        {overview_css}
        html,
        body {{
          margin: 0;
          background: transparent;
        }}
        .overview-reference {{
          margin: 0;
          padding: 0;
        }}
        </style>
        <section class="overview-reference">
          <div class="overview-hero">
            <div>
              <h1>Fraud Spike Detection</h1>
              <p>Identify unusual spikes in fraudulent transactions and take action early.</p>
            </div>
            <div class="overview-hero-pill">
              <span class="overview-shield">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.4 19.5 5.2v5.7c0 4.9-3 8.9-7.5 10.7-4.5-1.8-7.5-5.8-7.5-10.7V5.2L12 2.4Z"/><path d="M12 6.2v11.1c2.6-1.3 4.2-3.7 4.2-6.5V7.4L12 6.2Z"/></svg>
              </span>
              <div><strong>Detect <b>•</b> Analyze <b>•</b> Prevent</strong><span>Safer transactions. A more secure tomorrow.</span></div>
            </div>
          </div>

          <div class="overview-flow">
            <div class="overview-step-card">
              <div class="overview-step-head"><span>1</span><strong>Input Data</strong></div>
              <p>Upload or connect your transaction data.</p>
              <div class="overview-upload">
                <span class="overview-symbol">upload</span>
                <em>Drag & drop your file here<br>or</em>
                <button>Choose File</button>
              </div>
              <small>Supported formats: CSV, Excel</small>
            </div>
            <div class="overview-arrow">→</div>
            <div class="overview-step-card">
              <div class="overview-step-head"><span>2</span><strong>Detect Spikes</strong></div>
              <p>We analyze transaction patterns to find unusual spikes in fraud.</p>
              <div class="overview-icon-circle"><span class="overview-symbol">search</span></div>
              <ul><li>Monitor transaction volume</li><li>Identify abnormal patterns</li><li>Use statistical & ML models</li></ul>
            </div>
            <div class="overview-arrow">→</div>
            <div class="overview-step-card">
              <div class="overview-step-head"><span>3</span><strong>View Results</strong></div>
              <p>See detected spikes and high-risk transactions.</p>
              <div class="overview-icon-circle"><span class="overview-symbol">bar_chart</span></div>
              <ul><li>Time-wise fraud trend</li><li>Spike alerts</li><li>High-risk transactions list</li></ul>
            </div>
            <div class="overview-arrow">→</div>
            <div class="overview-step-card">
              <div class="overview-step-head"><span>4</span><strong>Take Action</strong></div>
              <p>Review and act on the detected fraud.</p>
              <div class="overview-icon-circle"><span class="overview-symbol">settings</span></div>
              <ul><li>Investigate suspicious activity</li><li>Block or flag transactions</li><li>Export report</li></ul>
            </div>
          </div>

          <div class="overview-analytics-row">
            <div class="overview-panel overview-trend-panel">
              <h2>Transaction Trend</h2>
              <div class="overview-legend"><span><i class="dot-muted"></i>Total Transactions</span><span><i></i>Fraudulent Transactions</span></div>
              <div class="overview-chart">
                <svg viewBox="0 0 470 245" preserveAspectRatio="none" aria-hidden="true">
                  <g class="overview-grid">
                    <path d="M36 28H450M36 76H450M36 124H450M36 172H450M36 220H450"/>
                    <path d="M36 28V220M105 28V220M174 28V220M243 28V220M312 28V220M381 28V220M450 28V220"/>
                  </g>
                  <polyline class="overview-line-muted" points="{points(total_points)}"/>
                  <polyline class="overview-line-blue" points="{points(flagged_points)}"/>
                  <circle class="overview-dot-blue" cx="446" cy="47" r="6"/>
                  <path class="overview-spike-arrow" d="M446 13v26m0 0-8-8m8 8 8-8"/>
                </svg>
                <div class="overview-spike-chip" style="left:{spike_label_x}px;top:{spike_label_y}px">Spike Detected</div>
                <div class="overview-axis-labels">{x_labels}</div>
              </div>
            </div>

            <div class="overview-panel overview-risk-table">
              <div class="overview-table-head"><h2>High-Risk Transactions</h2><span>View All</span></div>
              <table>
                <thead><tr><th>Transaction ID</th><th>Date & Time</th><th>Amount</th><th>Risk Score</th><th>Status</th></tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
          </div>

          <div class="overview-spike-callout">
            <div class="overview-callout-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.4 19.5 5.2v5.7c0 4.9-3 8.9-7.5 10.7-4.5-1.8-7.5-5.8-7.5-10.7V5.2L12 2.4Z"/><path d="M12 6.2v11.1c2.6-1.3 4.2-3.7 4.2-6.5V7.4L12 6.2Z"/></svg></div>
            <div class="overview-callout-copy"><h2>{escape(spike_title)}</h2><p>{escape(spike_copy)}</p></div>
            <div class="overview-callout-meta"><span>Spike Time</span><strong>{escape(latest_day)}, 10:00 - 12:00</strong></div>
            <div class="overview-callout-meta"><span>Increase</span><strong class="overview-increase">{escape(increase_label)}</strong></div>
            <button class="overview-investigate">Investigate Now</button>
          </div>

          <div class="overview-footer"><span>Fraud detection today. A safer tomorrow.</span><span>Secure <b>•</b> Reliable <b>•</b> Intelligent</span></div>
        </section>
        """),
        height=1190,
        scrolling=False,
    )


def _render_overview_trend_chart(transactions: pd.DataFrame) -> None:
    st.markdown("#### Transaction trend")
    daily = transactions.assign(day=transactions["created_at"].dt.floor("D"))
    trend = daily.groupby("day").agg(
        **{
            "Total transactions": ("payment_id", "count"),
            "Flagged high risk": ("risk_status", lambda status: status.eq("High risk").sum()),
        }
    )
    st.line_chart(trend, height=280)
    st.caption("Daily transaction volume vs. transactions flagged high risk in the loaded window.")


def _render_overview_high_risk_panel(alerts: pd.DataFrame) -> None:
    st.markdown("#### High-risk transactions")
    if alerts.empty:
        st.success("No transactions are at or above the review threshold in this window.")
        return
    reviewer = st.text_input(
        "Reviewer identity",
        placeholder="Name or work email",
        key="overview_reviewer_identity",
    )
    for _, row in alerts.head(5).iterrows():
        payment_id = str(row["payment_id"])
        score = float(row["risk_score"])
        is_high = score >= DEMO_BLOCKING_THRESHOLD
        status_label = "Flagged" if is_high else "Review"
        status_class = "high" if is_high else "review"
        created_at = pd.Timestamp(row["created_at"])
        with st.container(border=True):
            st.markdown(
                f'<span class="alert-card-anchor alert-card-{status_class}"></span>'
                f"""
                <div class="alert-card-header">
                  <span class="alert-priority alert-priority-{status_class}">{status_label}</span>
                  <span class="alert-risk-score"><strong>{score:.0%}</strong> risk score</span>
                </div>
                <div class="alert-payment-line">
                  <strong>{escape(str(row['currency']))} {float(row['amount']):,.2f}</strong>
                  <span>{escape(payment_id)}</span>
                </div>
                <div class="alert-payment-meta">{created_at:%d %b %Y, %H:%M UTC}</div>
                """,
                unsafe_allow_html=True,
            )
            legit_col, fraud_col, investigate_col = st.columns(3)
            with legit_col:
                mark_legit = st.button(
                    "Mark legitimate", key=f"overview_legit_{payment_id}", use_container_width=True
                )
            with fraud_col:
                mark_fraud = st.button(
                    "Confirm fraud", key=f"overview_fraud_{payment_id}", use_container_width=True
                )
            with investigate_col:
                st.button(
                    "Investigate →", key=f"overview_investigate_{payment_id}",
                    type="primary", use_container_width=True, on_click=_open_demo_view,
                    args=("Transaction investigation", payment_id),
                )
            if mark_legit or mark_fraud:
                try:
                    set_fraud_case_status(
                        payment_id,
                        "false_positive" if mark_legit else "confirmed_fraud",
                        SCORING_API_URL,
                        actor=reviewer,
                        risk_score=score,
                    )
                    st.success("Case updated.")
                    st.rerun()
                except ScoringAPIError as exc:
                    st.error(str(exc))
    if len(alerts) > 5:
        st.caption(f"Showing the top 5 of {len(alerts):,} flagged transactions—see all below.")


def _render_overview_spike_banner(transactions: pd.DataFrame) -> None:
    daily_flagged = (
        transactions.assign(day=transactions["created_at"].dt.floor("D"))
        .groupby("day")["risk_status"].apply(lambda status: status.eq("High risk").sum())
        .sort_index()
    )
    if len(daily_flagged) < 2:
        return
    latest_day, latest = daily_flagged.index[-1], daily_flagged.iloc[-1]
    baseline = daily_flagged.iloc[:-1].mean()
    if baseline <= 0 or latest <= baseline * 1.5:
        return
    increase = (latest / baseline - 1) * 100
    st.markdown(
        f"""
        <div class="spike-banner">
          <div><strong>Fraud spike detected!</strong><br>
          Transactions flagged high risk on {latest_day:%d %b %Y} are {increase:.0f}% above the
          window's daily average—see the high-risk queue above.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_alert_queue(alerts: pd.DataFrame) -> None:
    """The full, filterable list of every transaction at or above the review threshold."""
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


def render_overview(transactions: pd.DataFrame, *, connection: dict, is_mock_session: bool) -> None:
    """Reference-style fraud spike overview."""
    transactions, using_upload = active_transaction_dataset(transactions)
    if transactions.empty:
        render_page_header(
            "Fraud overview",
            "Detect unusual spikes, review flagged transactions, and act on them-all in one place.",
        )
        st.info("No transactions are loaded for the selected date range.")
    elif not transactions["risk_score"].notna().any():
        render_page_header(
            "Fraud overview",
            "Detect unusual spikes, review flagged transactions, and act on them-all in one place.",
        )
        st.info(
            "Fraud overview needs scored transactions. Real Razorpay payments have no FraudLens "
            "score because the Payments API does not supply the required signals."
        )
    else:
        if using_upload:
            st.info("Overview is using the uploaded CSV scored by the backend model.")
        alerts = transactions[transactions["risk_score"].ge(DEMO_REVIEW_THRESHOLD)].sort_values(
            ["risk_score", "created_at"], ascending=[False, False]
        )
        _render_overview_infographic(transactions, alerts)


def _render_transaction_chat(row: pd.Series) -> None:
    payment_id = str(row["payment_id"])
    chats = st.session_state.setdefault("razorpay_transaction_chats", {})
    history = chats.setdefault(payment_id, [])
    with st.container(height=420, border=True):
        if not history:
            st.caption("Ask why this transaction was flagged, summarize its risk factors, or request next steps.")
        for message in history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    question_col, send_col = st.columns([5, 1], vertical_alignment="bottom")
    with question_col:
        typed_question = st.text_input(
            "Ask about this transaction",
            key=f"investigation_question_{payment_id}",
            placeholder="Ask anything about this transaction…",
            label_visibility="collapsed",
        )
    with send_col:
        send_clicked = st.button(
            "Send question", key=f"investigation_send_{payment_id}", use_container_width=True
        )
    suggestions = (
        "Summarize risk factors",
        "Is this customer genuine?",
        "What should I do next?",
    )
    suggestion_columns = st.columns(3)
    selected_question = typed_question.strip() if send_clicked else ""
    for column, suggestion in zip(suggestion_columns, suggestions):
        with column:
            if st.button(
                suggestion,
                key=f"investigation_suggestion_{payment_id}_{suggestion}",
                use_container_width=True,
            ):
                selected_question = suggestion
    if send_clicked and not selected_question:
        st.warning("Enter a question before sending.")
        return
    if not selected_question:
        return

    prior = history[-8:]
    history.append({"role": "user", "content": selected_question})
    context = {
        "payment_id": payment_id, "transaction_id": payment_id,
        "timestamp": row["created_at"].isoformat(), "amount": float(row["amount"]),
        "currency": str(row["currency"]), "status": str(row["status"]),
        "method": str(row["method"]), "order_id": str(row["order_id"]),
        "email": str(row["email"]), "contact": str(row["contact"]),
        "international": bool(row["international"]),
        "velocity": None if pd.isna(row["velocity"]) else float(row["velocity"]),
        "ip_billing_mismatch": None if pd.isna(row["ip_billing_mismatch"]) else bool(row["ip_billing_mismatch"]),
        "new_device": None if pd.isna(row["new_device"]) else bool(row["new_device"]),
        "amount_deviation": None if pd.isna(row["amount_deviation"]) else float(row["amount_deviation"]),
        "risk_score": None if pd.isna(row["risk_score"]) else float(row["risk_score"]),
        "risk_status": None if pd.isna(row["risk_status"]) else str(row["risk_status"]),
        "actual": None if pd.isna(row["actual"]) else str(row["actual"]),
        "reasons": row.get("reasons") if isinstance(row.get("reasons"), list) else [],
        "review_threshold": float(
            (st.session_state.get("csv_tester_results") or {}).get("decision_threshold")
            or DEMO_REVIEW_THRESHOLD
        ),
    }
    try:
        response = ask_preview_transaction_question(
            context, selected_question, prior, SCORING_API_URL
        )
        answer = response["answer"] if response["status"] == "generated" else response.get("error", "Unable to answer.")
    except ScoringAPIError as exc:
        answer = str(exc)
    history.append({"role": "assistant", "content": answer})
    st.rerun()


def _step_investigation(options: list[str], delta: int) -> None:
    current = st.session_state.get("investigation_payment_id")
    if current in options:
        index = options.index(current)
        st.session_state["investigation_payment_id"] = options[max(0, min(len(options) - 1, index + delta))]


def _fetch_case(payment_id: str) -> tuple[dict | None, list[dict]]:
    try:
        payload = get_fraud_case(payment_id, SCORING_API_URL)
        return payload.get("case"), payload.get("notes") or []
    except ScoringAPIError as exc:
        st.warning(f"Case storage unavailable: {exc}")
        return None, []


def _render_investigation_notes_panel(payment_id: str, notes: list[dict]) -> None:
    st.markdown("### Investigation notes")
    if not notes:
        st.caption("No notes yet.")
    else:
        with st.container(height=220, border=False):
            for note in notes:
                st.markdown(
                    f'<div class="investigation-note"><strong>{escape(note.get("author") or "Analyst")}</strong> '
                    f'<span>{escape(str(note.get("created_at", "")))}</span>'
                    f'<p>{escape(note["note"])}</p></div>',
                    unsafe_allow_html=True,
                )
    note_text = st.text_area(
        "Add a note", key=f"case_note_{payment_id}", placeholder="Investigation notes…",
        label_visibility="collapsed",
    )
    if st.button("+ Add note", key=f"case_save_note_{payment_id}", use_container_width=True):
        analyst = st.session_state.get("investigation_analyst_name", "")
        if note_text.strip():
            try:
                add_fraud_case_note(payment_id, note_text, SCORING_API_URL, author=analyst)
                st.success("Note saved.")
                st.rerun()
            except ScoringAPIError as exc:
                st.error(str(exc))
        else:
            st.warning("Write a note before saving.")


def _render_case_status_bar(row: pd.Series, case: dict | None) -> None:
    payment_id = str(row["payment_id"])
    current_status = case["status"] if case else "open"
    updated_by = f" · last updated by {case['updated_by']}" if case and case.get("updated_by") else ""
    st.markdown(
        f"Case status: **{CASE_STATUS_LABELS.get(current_status, current_status)}**{updated_by}"
    )
    analyst = st.text_input(
        "Analyst name", key="investigation_analyst_name", placeholder="Name or work email"
    )
    legit_col, review_col, escalate_col = st.columns(3)
    with legit_col:
        mark_false_positive = st.button(
            "Mark as legitimate", use_container_width=True, key=f"case_fp_{payment_id}"
        )
    with review_col:
        mark_investigating = st.button(
            "Keep under investigation", use_container_width=True, key=f"case_investigating_{payment_id}"
        )
    with escalate_col:
        mark_fraud = st.button(
            "Escalate: confirm fraud", type="primary", use_container_width=True, key=f"case_fraud_{payment_id}"
        )
    action = (
        "false_positive" if mark_false_positive
        else "under_investigation" if mark_investigating
        else "confirmed_fraud" if mark_fraud
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


def render_transaction_investigation(transactions: pd.DataFrame, *, is_mock: bool) -> None:
    """Full signal detail, the AI evidence report, grounded chat, and case actions for one transaction."""
    st.markdown('<span class="investigation-reference-active"></span>', unsafe_allow_html=True)
    render_page_header(
        "Transaction investigation",
        "Review transaction details, analyze risk factors, and take appropriate action.",
    )
    transactions, using_upload = active_transaction_dataset(transactions)
    if transactions.empty:
        st.info("No transactions are loaded for the selected date range.")
        return
    options = transactions.sort_values(
        "risk_score", ascending=False, na_position="last"
    )["payment_id"].astype(str).tolist()
    if st.session_state.get("investigation_payment_id") not in options:
        st.session_state["investigation_payment_id"] = options[0]
    current_index = options.index(st.session_state["investigation_payment_id"])
    payment_id = st.session_state["investigation_payment_id"]
    row = transactions.loc[transactions["payment_id"].astype(str).eq(payment_id)].iloc[0]
    scored = not pd.isna(row["risk_score"])

    if scored:
        status_label = str(row["risk_status"])
        badge_class = "high" if status_label == "High risk" else "review" if status_label == "Review" else "safe"
        badge_html = f'<span class="alert-priority alert-priority-{badge_class}">{escape(status_label)}</span>'
    else:
        badge_html = ""
    amount_label = f"{escape(str(row['currency']))} {float(row['amount']):,.2f}"
    score_label = f"{float(row['risk_score']) * 100:.0f} / 100" if scored else "N/A"
    status_chip = escape(str(row["risk_status"] if scored else row["status"]).replace("_", " ").title())
    method_label = str(row["method"]).title()
    customer_label = str(row["email"]) or str(row["contact"]) or "Customer"
    detail_pairs = [
        ("Transaction ID", payment_id),
        ("Date & Time", f"{row['created_at']:%b %d, %Y, %I:%M %p}"),
        ("Payment Method", method_label),
        ("Order ID", str(row.get("order_id") or "Not available")),
        ("Customer", customer_label),
        ("Currency", str(row["currency"])),
        ("International", "Yes" if bool(row["international"]) else "No"),
        (
            "Location comparison",
            "Not available" if pd.isna(row["ip_billing_mismatch"])
            else "Mismatch" if bool(row["ip_billing_mismatch"]) else "Match",
        ),
        (
            "Device history",
            "Not available" if pd.isna(row["new_device"])
            else "New device" if bool(row["new_device"]) else "Known device",
        ),
        ("Recent velocity", "Not available" if pd.isna(row["velocity"]) else f"{float(row['velocity']):.1f}/hr"),
        (
            "Amount deviation",
            "Not available" if pd.isna(row["amount_deviation"])
            else f"{float(row['amount_deviation']):+.0f}% vs. baseline",
        ),
        ("Payment Status", str(row["status"]).replace("_", " ").title()),
        ("Transaction Status", status_chip),
        ("Risk Score", score_label),
    ]
    details_html = "".join(
        f'<div><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>'
        for label, value in detail_pairs
    )
    severity_badge = {"high": "High", "medium": "Medium", "low": "Low"}
    risk_cards = []
    if scored and not any(pd.isna(row[field]) for field in (
        "velocity", "ip_billing_mismatch", "new_device", "amount_deviation"
    )):
        risk_cards = [
            (card["icon"], card["title"], card["detail"], card["severity"])
            for card in _risk_indicator_cards(row)
        ]
    cards_html = "".join(
        f"""
        <div class="ti-risk-card ti-risk-{severity}">
          <span class="material-symbols-rounded">{icon}</span>
          <strong>{title}</strong>
          <p>{detail}</p>
          <em>{severity_badge[severity]}</em>
        </div>
        """
        for icon, title, detail, severity in risk_cards
    ) or '<p class="ti-empty-state">No model risk indicators are available for this transaction.</p>'
    related = transactions.loc[
        ~transactions["payment_id"].astype(str).eq(payment_id)
    ].sort_values(["risk_score", "created_at"], ascending=[False, False], na_position="last").head(4)
    related_rows = []
    for _, item in related.iterrows():
        related_score = "N/A" if pd.isna(item["risk_score"]) else f"{float(item['risk_score']) * 100:.0f}"
        related_status = str(item.get("risk_status") or item["status"]).replace("_", " ").title()
        related_pill = "review" if str(item.get("risk_status")) == "Review" else "flagged"
        related_rows.append(f"""
        <tr>
          <td>{escape(str(item['payment_id']))}</td>
          <td>{item['created_at']:%b %d, %I:%M %p}</td>
          <td>{escape(str(item['currency']))} {float(item['amount']):,.2f}</td>
          <td>{escape(str(item['method']).title())}</td>
          <td>{related_score}</td>
          <td><span class="ti-table-pill {related_pill}">{escape(related_status)}</span></td>
        </tr>
        """)
    related_html = "".join(related_rows)
    back_col, nav_spacer, previous_col, next_col = st.columns(
        [2.4, 4, 1.5, 1.5], vertical_alignment="center"
    )
    with back_col:
        st.button(
            "← Back to alerts", key="investigation_back", on_click=_open_demo_view,
            args=("Overview",), use_container_width=True,
        )
    with previous_col:
        st.button(
            "Previous transaction", key="investigation_previous", disabled=current_index == 0,
            on_click=_step_investigation, args=(options, -1), use_container_width=True,
        )
    with next_col:
        st.button(
            "Next transaction", key="investigation_next", disabled=current_index == len(options) - 1,
            on_click=_step_investigation, args=(options, 1), use_container_width=True,
        )

    main_col, assistant_col = st.columns([1.42, 1], gap="large", vertical_alignment="top")
    with main_col:
        st.markdown(
            f"""
        <div class="ti-shell">
          <div class="ti-main-stack">
            <main>
              <section class="ti-hero">
                <div class="ti-icon"><span class="material-symbols-rounded">credit_card</span></div>
                <div>
                  <span>Transaction ID</span>
                  <h2>{escape(payment_id)}</h2>
                  <p>{row['created_at']:%b %d, %Y, %I:%M %p}</p>
                </div>
                {badge_html}
                <div class="ti-amount"><span>Amount</span><strong>{amount_label}</strong></div>
              </section>
              <section class="ti-card">
                <div class="ti-card-head"><h3>Transaction Details</h3></div>
                <div class="ti-detail-grid">{details_html}</div>
              </section>
              <section class="ti-card">
                <div class="ti-card-head"><h3>Risk Indicators</h3></div>
                <div class="ti-risk-grid">{cards_html}</div>
              </section>
              <section class="ti-card">
                <div class="ti-card-head"><h3>Related Transactions</h3></div>
                <table class="ti-related">
                  <thead><tr><th>Transaction ID</th><th>Date & Time</th><th>Amount</th><th>Payment method</th><th>Risk Score</th><th>Status</th></tr></thead>
                  <tbody>{related_html}</tbody>
                </table>
              </section>
            </main>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with assistant_col:
        with st.container(border=True):
            st.markdown('<span class="ti-interactive-ai-anchor"></span>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ti-ai-title"><span class="material-symbols-rounded">auto_awesome</span>'
                '<div><h3>Ask About This Transaction</h3>'
                '<p>Answers come from the backend using this transaction’s recorded risk evidence.</p>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            _render_transaction_chat(row)
            if scored:
                score = float(row["risk_score"])
                if st.button(
                    "Generate full evidence report", type="primary",
                    key="investigation_generate_report", use_container_width=True,
                ):
                    transaction = {
                        "payment_id": payment_id, "velocity": float(row["velocity"]),
                        "ip_billing_mismatch": bool(row["ip_billing_mismatch"]),
                        "new_device": bool(row["new_device"]),
                        "amount_deviation": float(row["amount_deviation"]),
                        "risk_score": score,
                    }
                    try:
                        with st.spinner("Turning verified signals into a reviewer-ready explanation…"):
                            report = generate_demo_transaction_report(
                                transaction, SCORING_API_URL, threshold=DEMO_BLOCKING_THRESHOLD
                            )
                        st.session_state["demo_evidence_report"] = {
                            "payment_id": payment_id, "report": report
                        }
                    except ScoringAPIError as exc:
                        st.error(str(exc))
                saved = st.session_state.get("demo_evidence_report")
                if saved and saved["payment_id"] == payment_id:
                    report = saved["report"]
                    if report.get("summary"):
                        st.info(report["summary"])
                    elif report.get("error"):
                        st.warning(report["error"])

        with st.container(border=True):
            st.markdown('<span class="ti-interactive-notes-anchor"></span>', unsafe_allow_html=True)
            case, notes = _fetch_case(payment_id)
            _render_investigation_notes_panel(payment_id, notes)
            _render_case_status_bar(row, case)

CSV_TESTER_REQUIRED_COLUMNS = [
    "transaction_id", "timestamp", "user_id", "device_id", "card_id",
    "amount", "billing_country", "ip_country", "merchant_category",
]


def _render_csv_batch_tester() -> None:
    """Score an analyst-supplied CSV with the real trained model, independent of the demo data above."""
    template = pd.DataFrame([{
        "transaction_id": "txn_0001",
        "timestamp": "2026-01-15T09:30:00Z",
        "user_id": "user_123",
        "device_id": "device_abc",
        "card_id": "card_xyz",
        "amount": 4999.00,
        "billing_country": "IN",
        "ip_country": "IN",
        "merchant_category": "electronics",
    }])
    with st.container(border=True):
        st.markdown('<span class="csv-tester-anchor"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="csv-tester-title"><span class="material-symbols-rounded">upload_file</span>'
            'Run the fraud model on a CSV</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="csv-tester-caption">Attach your transaction data, then run the trained '
            'model to identify suspicious transactions and review their fraud scores.</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Attach transactions CSV", type=["csv"], key="csv_tester_upload"
        )
        run_clicked = st.button(
            "Run model", type="primary", key="csv_tester_score_button", use_container_width=True
        )
        st.download_button(
            "Download CSV template", template.to_csv(index=False), "fraud_test_template.csv",
            mime="text/csv", key="csv_tester_template", use_container_width=True,
        )
        st.caption(
            "Auto-detects uploaded CSV columns. Model fields: "
            + ", ".join(CSV_TESTER_REQUIRED_COLUMNS)
        )
        if uploaded is None:
            if run_clicked:
                st.info("Attach a CSV file before running the model.")
            return
        try:
            uploaded_df = pd.read_csv(uploaded)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not read this file as CSV: {exc}")
            return
        if uploaded_df.empty:
            st.info("The uploaded CSV has no rows to score.")
            return
        try:
            normalized_upload = normalize_uploaded_transactions(uploaded_df)
        except (TypeError, ValueError) as exc:
            st.error(f"Could not detect usable transaction values: {exc}")
            return
        uploaded_transactions = normalized_upload["transactions"]
        mapped = normalized_upload["mapped"]
        inferred = normalized_upload["inferred"]
        ignored = normalized_upload["ignored"]
        st.success(
            f"Detected {len(mapped)} model field(s) from {len(uploaded_df.columns)} CSV column(s). "
            f"{len(uploaded_transactions):,} row(s) ready to score."
        )
        if mapped:
            st.caption(
                "Detected columns: "
                + ", ".join(f"{source} → {target}" for source, target in mapped.items())
            )
        if inferred:
            st.warning(
                "Missing model fields were inferred: " + ", ".join(inferred)
                + ". Scores use the available evidence and may have lower confidence."
            )
        if ignored:
            st.caption("Additional CSV columns retained by the source but not used by this model: " + ", ".join(ignored))
        if run_clicked:
            try:
                with st.spinner(
                    f"Scoring and saving {len(uploaded_transactions):,} transaction(s) with the trained model…"
                ):
                    saved_dataset = score_and_save_uploaded_dataset(
                        uploaded_transactions, uploaded.name, SCORING_API_URL
                    )
                st.session_state["csv_tester_results"] = {
                    "file_id": uploaded.file_id,
                    **saved_dataset,
                }
                st.session_state["csv_tester_active"] = True
                for key in (
                    "explorer_filter_search", "explorer_filter_status", "explorer_filter_method",
                    "explorer_filter_currency", "explorer_filter_risk_status", "explorer_filter_international",
                    "explorer_geography_filter", "explorer_device_filter", "explorer_amount_min",
                    "explorer_amount_max", "explorer_risk_min", "explorer_risk_max",
                ):
                    st.session_state.pop(key, None)
                st.rerun()
            except (ScoringAPIError, ValueError) as exc:
                st.error(str(exc))
                st.session_state.pop("csv_tester_results", None)
                st.session_state.pop("csv_tester_active", None)

        cached = st.session_state.get("csv_tester_results")
        if cached and cached["file_id"] == uploaded.file_id:
            scored = cached["scored"]
            if cached.get("storage_status") == "unavailable":
                st.warning(
                    f"Scored {cached['row_count']:,} rows with the trained model. "
                    f"Database save is pending: {cached.get('storage_error')}"
                )
            else:
                st.success(
                    f"Scored and saved {cached['row_count']:,} rows. Dataset ID: {cached['dataset_id']}"
                )
            flagged = int(scored["flagged"].sum())
            flag_rate = flagged / len(scored) if len(scored) else 0.0
            r1, r2, r3 = st.columns(3)
            r1.metric("Transactions scored", f"{len(scored):,}")
            r2.metric("Flagged by the model", f"{flagged:,}")
            r3.metric("Flag rate", f"{flag_rate:.1%}")

            def result_series(column: str, default: object = None) -> pd.Series:
                if column in scored.columns:
                    return scored[column]
                return pd.Series([default] * len(scored), index=scored.index)

            currency = result_series("currency", "INR").fillna("INR").astype(str).str.upper()
            amount = pd.to_numeric(result_series("amount"), errors="coerce")
            velocity = pd.to_numeric(result_series("velocity"), errors="coerce")
            if velocity.isna().all():
                velocity = pd.to_numeric(result_series("card_txn_count_1h"), errors="coerce")
            ip_billing = result_series("ip_billing")
            computed_ip_billing = result_series("geo_mismatch").map(
                {1.0: "Mismatch", 0.0: "Match", True: "Mismatch", False: "Match"}
            )
            ip_billing = ip_billing.where(ip_billing.notna(), computed_ip_billing).fillna("—")
            device = result_series("device")
            computed_device = result_series("is_new_device").map(
                {1.0: "New", 0.0: "Known", True: "New", False: "Known"}
            )
            device = device.where(device.notna(), computed_device).fillna("—")
            amount_deviation = pd.to_numeric(
                result_series("amount_deviation"), errors="coerce"
            )
            if amount_deviation.isna().all():
                amount_deviation = pd.to_numeric(
                    result_series("user_amount_zscore"), errors="coerce"
                )
            hour = pd.to_numeric(result_series("hour"), errors="coerce")
            if hour.isna().all():
                hour = pd.to_datetime(result_series("timestamp"), utc=True, errors="coerce").dt.hour
            timestamp = pd.to_datetime(
                result_series("timestamp"), utc=True, errors="coerce"
            ).dt.strftime("%d %b %Y, %H:%M")
            status = result_series("status").where(
                result_series("status").notna(),
                scored["flagged"].map({True: "Flagged", False: "Not flagged"}),
            )
            actual = result_series("actual").fillna("—")
            display = pd.DataFrame({
                "Txn": scored["transaction_id"],
                "Timestamp": timestamp,
                "Amount (₹)": [
                    f"{'₹' if code == 'INR' else code + ' '}{value:,.2f}"
                    if pd.notna(value) else "—"
                    for code, value in zip(currency, amount, strict=True)
                ],
                "Velocity": velocity.round(2),
                "IP/billing": ip_billing,
                "Device": device,
                "Amt. dev.": amount_deviation.round(2),
                "Hour": hour.round().astype("Int64"),
                "Score": scored["score"].astype(float),
                "Status": status,
                "Actual": actual,
                "Reasons": scored["reasons"].map(
                    lambda items: "; ".join(items) if isinstance(items, list) else str(items)
                ),
            })
            st.caption(
                "Velocity, IP/billing, device, amount deviation, hour, score, and status come from the "
                "backend scoring response. Actual is shown only when the uploaded CSV includes a label."
            )
            st.dataframe(
                display, use_container_width=True, hide_index=True,
                column_config={
                    "Velocity": st.column_config.NumberColumn("Velocity", format="%.2f"),
                    "Amt. dev.": st.column_config.NumberColumn("Amt. dev.", format="%.2f"),
                    "Hour": st.column_config.NumberColumn("Hour", format="%02d"),
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                },
            )
            st.download_button(
                "Download scored results", csv_injection_safe(display).to_csv(index=False),
                "scored_transactions.csv", mime="text/csv", key="csv_tester_download",
            )


def render_case_management(transactions: pd.DataFrame) -> None:
    """Analyst case-management workbench, backed by the real case-store API.

    Cases live in fraud_cases/fraud_case_notes (backend/src/case_store.py) keyed
    only by transaction_id, status, risk_score, and timestamps — there is no
    customer/amount/priority field there. Those columns are enriched here by
    joining against whichever transactions are currently loaded (mock demo data
    or a connected Razorpay Test Mode account), the same dataset Transaction
    explorer/investigation use, so a case looks up its own transaction's details
    rather than duplicating them.
    """
    st.markdown('<span class="case-management-route-label">Case management</span>', unsafe_allow_html=True)
    render_page_header(
        "Case Management",
        "View, investigate, and resolve fraud cases. Track status, assign ownership, and take action.",
    )
    try:
        cases = list_fraud_cases(SCORING_API_URL)
    except ScoringAPIError as exc:
        st.warning(f"Case storage unavailable: {exc}")
        return
    if not cases:
        st.info(
            "No cases have been opened yet. Marking a transaction under investigation, "
            "confirmed fraud, or a false positive from Transaction investigation opens one here."
        )
        return

    cases_df = pd.DataFrame(cases)
    cases_df["created_at"] = pd.to_datetime(cases_df["created_at"], errors="coerce", utc=True)
    cases_df["updated_at"] = pd.to_datetime(cases_df["updated_at"], errors="coerce", utc=True)

    txn_lookup = (
        transactions.set_index("payment_id") if not transactions.empty else pd.DataFrame()
    )

    def _priority(txn_id: str, fallback_score: float | None) -> str:
        if txn_id in txn_lookup.index:
            return RISK_STATUS_PILL_LABELS.get(txn_lookup.loc[txn_id, "risk_status"], "—")
        if fallback_score is None or pd.isna(fallback_score):
            return "—"
        if fallback_score >= DEMO_BLOCKING_THRESHOLD:
            return "Flagged"
        if fallback_score >= DEMO_REVIEW_THRESHOLD:
            return "Review"
        return "Legit"

    cases_df["priority"] = [
        _priority(txn_id, score)
        for txn_id, score in zip(cases_df["transaction_id"], cases_df["risk_score"])
    ]
    cases_df["customer"] = cases_df["transaction_id"].map(
        lambda txn_id: (
            (str(txn_lookup.loc[txn_id, "email"]) or str(txn_lookup.loc[txn_id, "contact"]))
            if txn_id in txn_lookup.index else "—"
        )
    )
    cases_df["amount_label"] = cases_df["transaction_id"].map(
        lambda txn_id: (
            f"{txn_lookup.loc[txn_id, 'currency']} {txn_lookup.loc[txn_id, 'amount']:,.2f}"
            if txn_id in txn_lookup.index else "—"
        )
    )

    total = len(cases_df)
    open_count = int((cases_df["status"] == "open").sum())
    resolved_mask = cases_df["status"].isin(["confirmed_fraud", "false_positive"])
    resolved_count = int(resolved_mask.sum())
    resolved_deltas = (
        cases_df.loc[resolved_mask, "updated_at"] - cases_df.loc[resolved_mask, "created_at"]
    ).dropna()
    if not resolved_deltas.empty:
        avg_seconds = resolved_deltas.dt.total_seconds().mean()
        hours, minutes = divmod(int(avg_seconds // 60), 60)
        avg_label = f"{hours}h {minutes}m" if hours else f"{minutes}m"
    else:
        avg_label = "—"

    st.markdown(
        '<div class="cm-metrics">'
        f'<div class="cm-metric"><span class="cm-metric-icon blue">folder</span>'
        f'<div><small>Total cases</small><strong>{total:,}</strong></div></div>'
        f'<div class="cm-metric"><span class="cm-metric-icon red">warning</span>'
        f'<div><small>Open cases</small><strong>{open_count:,}</strong></div></div>'
        f'<div class="cm-metric"><span class="cm-metric-icon green">check_circle</span>'
        f'<div><small>Resolved cases</small><strong>{resolved_count:,}</strong></div></div>'
        f'<div class="cm-metric"><span class="cm-metric-icon purple">schedule</span>'
        f'<div><small>Avg. resolution time</small><strong>{avg_label}</strong></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    filter_cols = st.columns(4)
    with filter_cols[0]:
        status_filter = st.multiselect(
            "Status", list(CASE_STATUS_LABELS.values()), key="cm_filter_status",
        )
    with filter_cols[1]:
        priority_filter = st.multiselect(
            "Priority", ["Flagged", "Review", "Legit"], key="cm_filter_priority",
        )
    with filter_cols[2]:
        assignee_options = sorted(v for v in cases_df["updated_by"].dropna().unique() if v)
        assignee_filter = st.multiselect("Assigned to", assignee_options, key="cm_filter_assignee")
    with filter_cols[3]:
        search = st.text_input(
            "Search", placeholder="Transaction ID, customer…", key="cm_filter_search",
        )

    filtered = cases_df.reset_index(drop=True)
    if status_filter:
        keys = {key for key, label in CASE_STATUS_LABELS.items() if label in status_filter}
        filtered = filtered[filtered["status"].isin(keys)]
    if priority_filter:
        filtered = filtered[filtered["priority"].isin(priority_filter)]
    if assignee_filter:
        filtered = filtered[filtered["updated_by"].isin(assignee_filter)]
    if search:
        needle = search.strip().lower()
        filtered = filtered[
            filtered["transaction_id"].str.lower().str.contains(needle, na=False)
            | filtered["customer"].str.lower().str.contains(needle, na=False)
        ]
    filtered = filtered.reset_index(drop=True)

    list_col, detail_col = st.columns([2.1, 1.15], gap="large")
    with list_col:
        st.markdown(f"#### Cases ({len(filtered)})")
        if filtered.empty:
            st.info("No cases match these filters.")
        else:
            display = pd.DataFrame({
                "Transaction ID": filtered["transaction_id"],
                "Customer": filtered["customer"],
                "Amount": filtered["amount_label"],
                "Risk score": filtered["risk_score"],
                "Priority": filtered["priority"],
                "Status": filtered["status"].map(CASE_STATUS_LABELS),
                "Assigned to": filtered["updated_by"].fillna("Unassigned"),
                "Updated": filtered["updated_at"],
            })
            event = st.dataframe(
                display,
                hide_index=True,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="cm_case_table",
                column_config={
                    "Risk score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                    "Updated": st.column_config.DatetimeColumn(format="D MMM, h:mm a"),
                },
            )
            selected_rows = event.selection.rows if event and event.selection else []
            if selected_rows:
                st.session_state["cm_selected_case"] = filtered.iloc[selected_rows[0]]["transaction_id"]
            elif "cm_selected_case" not in st.session_state:
                st.session_state["cm_selected_case"] = filtered.iloc[0]["transaction_id"]

    with detail_col:
        selected_id = st.session_state.get("cm_selected_case")
        if not selected_id or selected_id not in filtered["transaction_id"].values:
            st.info("Select a case from the list to inspect it.")
        else:
            case_row = cases_df.loc[cases_df["transaction_id"] == selected_id].iloc[0]
            case, notes = _fetch_case(selected_id)
            with st.container(border=True):
                st.markdown(f"##### {escape(selected_id)}")
                st.caption(
                    f"Priority: {case_row['priority']} · "
                    f"{CASE_STATUS_LABELS.get(case_row['status'], case_row['status'])}"
                )
                if selected_id in txn_lookup.index:
                    txn_row = txn_lookup.loc[selected_id]
                    st.markdown(
                        f"**Amount:** {txn_row['currency']} {txn_row['amount']:,.2f}  \n"
                        f"**Customer:** {escape(str(txn_row.get('email') or txn_row.get('contact') or '—'))}  \n"
                        f"**Method:** {escape(str(txn_row.get('method', '—')))}"
                    )
                fallback_row = pd.Series({
                    "payment_id": selected_id,
                    "risk_score": case_row.get("risk_score", float("nan")),
                })
                _render_case_status_bar(fallback_row, case)
                _render_investigation_notes_panel(selected_id, notes)


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
        "Overview": "monitoring",
        "Transaction explorer": "travel_explore",
        "Transaction investigation": "manage_search",
        "Case management": "folder_open",
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

def _run_global_search() -> None:
    """Route the top-bar search box into Transaction explorer's search filter."""
    query = st.session_state.get("global_search_query", "").strip()
    if query:
        st.session_state["fraudlens_view"] = "Transaction explorer"
        st.session_state["explorer_search_prefill"] = query


account_id = connection.get("razorpay_account_id", "")
connection_name = "Demo mode" if is_mock_session else "Razorpay connected"
connection_detail = (
    "Synthetic UI fixtures · rule-based demo scores" if is_mock_session else account_id
)
flagged_count = sum(
    1 for payment in st.session_state.get("razorpay_payments", [])
    if payment.get("risk_status") == "High risk"
)
avatar_label = "Poojitha"
avatar_initials = "PP"

if view == "Overview":
    st.markdown(
        '<style>.block-container{max-width:none!important;padding-left:1.55rem!important;'
        'padding-right:1.55rem!important;}</style>'
        '<span class="overview-screen-active"></span>'
        '<span class="overview-route-label">Fraud Spike Detection Transaction Trend '
        'Fraud Spike Detected! Detect Spikes High-Risk Transactions</span>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="topnav-state-text">{connection_name} · {connection_detail}</div>',
    unsafe_allow_html=True,
)
brand_col, search_col, bell_col, account_col, *disconnect_col = st.columns(
    [2.25, 3.2, 0.36, 1.05] + ([0.86] if not is_mock_session else []),
    vertical_alignment="center",
)
with brand_col:
    st.markdown(
        fraudguard_brand_lockup(),
        unsafe_allow_html=True,
    )
with search_col:
    st.text_input(
        "Search",
        key="global_search_query",
        placeholder="Search cases, transaction IDs, customer IDs...",
        label_visibility="collapsed",
        on_change=_run_global_search,
    )
with bell_col:
    st.markdown(
        f'<span class="topbar-bell-anchor" data-count="{flagged_count if flagged_count else 3}"></span>',
        unsafe_allow_html=True,
    )
    st.button(
        "",
        icon=":material/notifications:",
        key="topbar_bell", help="Flagged transactions in the loaded window",
        on_click=_open_demo_view, args=("Overview",),
    )
with account_col:
    st.markdown(
        f'<div class="topbar-account"><div class="topbar-avatar">{avatar_initials}</div>'
        f'<strong>{avatar_label}</strong><span class="topbar-chevron">⌄</span></div>',
        unsafe_allow_html=True,
    )
disconnect = False
if not is_mock_session and disconnect_col:
    with disconnect_col[0]:
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

if view == "Overview":
    transactions = load_dashboard_transactions(show_date_filter=False)
    st.markdown('<div class="overview-content-spacer"></div>', unsafe_allow_html=True)
    render_overview(transactions, connection=connection, is_mock_session=is_mock_session)
elif view == "Case management":
    transactions = load_dashboard_transactions(show_date_filter=False)
    render_case_management(transactions)
else:
    transactions = load_dashboard_transactions(show_date_filter=view == "Transaction explorer")
    if transactions.empty:
        st.info("No transactions were found for the selected date range.")
    elif view == "Transaction explorer":
        render_transactions_view(transactions, is_mock=is_mock_session)
    elif view == "Transaction investigation":
        render_transaction_investigation(transactions, is_mock=is_mock_session)
