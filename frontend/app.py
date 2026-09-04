"""FraudLens transaction review dashboard."""

from __future__ import annotations

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
    ask_preview_transaction_question,
    ask_transaction_question,
    cheapest_threshold_row,
    cost_curve_for_ratio,
    csv_injection_safe,
    demo_case_catalog,
    generate_demo_transaction_report,
    load_global_importance,
    load_threshold_curve,
    risk_audit_rows,
    risk_evidence_summary,
    score_uploaded_transactions,
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


def brand_lockup(size: str = "26px", text_size: str = "1.05rem", text_color: str = "#eef4ff", stroke: str = "#3395ff") -> str:
    mark = CUT_LENS_MARK_SVG.format(stroke=stroke)
    return (
        f'<div class="brand-lockup" style="font-size:{text_size};color:{text_color};">'
        f'<span class="brand-mark" style="width:{size};height:{size};">{mark}</span>'
        f'<span>Fraud<span style="color:{stroke};">Lens</span></span></div>'
    )


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=Inter:wght@400;500;600&display=swap');
    :root {
      --navy: #0c2451; --navy-2: #123168; --blue: #3395ff; --blue-hover: #1c7ae6;
      --blue-light: #8ec5ff; --bg: #f5f8ff; --border: #dbe6fa; --panel-bg: #eaf2ff;
      --text-muted: #55698c; --text-dark: #0f2247; --sidebar-card: #14294f; --sidebar-border: #234a7a;
    }
    body, .stApp { font-family: 'Inter', system-ui, sans-serif; }
    .stApp { background: var(--bg); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: var(--navy); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p { color: #eef4ff; }
    .block-container { max-width: 1420px; padding-top: 2.2rem; padding-bottom: 4rem; }
    .hero { padding: 1.8rem 2rem; border-radius: 20px; color: white;
      background: linear-gradient(120deg, var(--navy) 0%, var(--navy-2) 58%, var(--blue) 100%);
      box-shadow: 0 16px 42px rgba(12,36,81,.18); margin-bottom: 1.35rem; }
    .hero-kicker { color: var(--blue-light); font-size: .78rem; font-weight: 800; letter-spacing: .12em; }
    .hero h1 { font-size: 2.25rem; margin: .35rem 0 .55rem; }
    .hero p { color: #d9e6f5; font-size: 1.05rem; max-width: 860px; margin: 0; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--border); padding: .9rem 1rem;
      border-radius: 14px; box-shadow: 0 3px 12px rgba(12,36,81,.05); }
    [data-testid="stMetricLabel"] p { color: var(--text-muted) !important; }
    [data-testid="stMetricValue"] { color: var(--text-dark) !important; }
    [data-testid="stBaseButton-primary"] { background: var(--blue); border-color: var(--blue); }
    [data-testid="stBaseButton-primary"]:hover { background: var(--blue-hover); border-color: var(--blue-hover); }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
      border-color: var(--sidebar-border); background: var(--sidebar-card); color: white; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    .workflow { display:grid; grid-template-columns:repeat(3,1fr); margin:0 0 1.4rem;
      border:1px solid var(--border); border-radius:4px; overflow:hidden; background:white; }
    .workflow.workflow-four { grid-template-columns:repeat(4,1fr); }
    .workflow-step { padding:1rem 1.15rem; border-right:1px solid var(--border); background:var(--panel-bg); }
    .workflow-step:last-child { border-right:0; background:var(--navy-2); color:white; }
    .workflow-number { color:var(--blue); font-family:monospace; font-size:.82rem; margin-bottom:.3rem; }
    .workflow-step:last-child .workflow-number { color:var(--blue-light); }
    .workflow-title { font-weight:750; font-size:.98rem; color: var(--text-dark); }
    .workflow-step:last-child .workflow-title { color: white; }
    .importance-panel { border:1px solid var(--border); border-radius:5px; background:white; margin:1rem 0 1.35rem; }
    .importance-title { padding:1rem 1.15rem; border-bottom:1px solid var(--border); font-weight:750; color: var(--text-dark); }
    .importance-row { display:grid; grid-template-columns:270px 1fr 55px; gap:1rem; align-items:center;
      padding:.7rem 1.15rem; border-bottom:1px solid var(--border); color: var(--text-dark); }
    .importance-row:last-child { border-bottom:0; }
    .importance-track { height:10px; border-radius:3px; background:var(--panel-bg); overflow:hidden; }
    .importance-fill { height:100%; background:var(--blue); }
    .report-spotlight { padding:1.25rem 1.35rem; border:2px solid var(--blue); border-radius:12px;
      background:linear-gradient(135deg,#eef6ff,#ffffff); box-shadow:0 10px 28px rgba(51,149,255,.12); }
    .report-kicker { color:var(--blue-hover); font-size:.76rem; font-weight:800; letter-spacing:.1em; }
    .report-spotlight h2 { color:var(--text-dark); margin:.25rem 0 .35rem; font-size:1.45rem; }
    .report-spotlight p { color:var(--text-muted); margin:0; line-height:1.5; }
    .policy-spotlight { padding:1.25rem 1.35rem; border:2px solid #d9a441; border-radius:12px;
      background:linear-gradient(135deg,#fff8ea,#ffffff); box-shadow:0 10px 28px rgba(217,164,65,.14); }
    .policy-kicker { color:#a5731a; font-size:.76rem; font-weight:800; letter-spacing:.1em; }
    .policy-spotlight h2 { color:var(--text-dark); margin:.25rem 0 .35rem; font-size:1.45rem; }
    .policy-spotlight p { color:var(--text-muted); margin:0; line-height:1.5; }
    .defense-banner { padding:.6rem 1rem; border:1px solid #2f8f5b; border-radius:8px;
      background:#eafbf1; color:#1c6b41; font-size:.85rem; margin:.75rem 0; }
    .mock-enforcement-shell { border:3px solid #7b4ce2; border-radius:16px; padding:1.25rem 1.4rem;
      margin:1rem 0 1.3rem; background:linear-gradient(135deg,#f5f0ff,#ffffff);
      box-shadow:0 12px 30px rgba(74,43,140,.13); }
    .mock-enforcement-kicker { color:#6836cf; font-size:.78rem; font-weight:800; letter-spacing:.12em; }
    .mock-enforcement-shell h2 { color:var(--text-dark); margin:.3rem 0 .35rem; }
    .mock-enforcement-shell p { color:var(--text-muted); margin:0; line-height:1.5; }
    .demo-tour { padding:1.35rem 1.45rem; border:2px solid var(--accent); border-radius:12px;
      margin:0 0 1.35rem; background:linear-gradient(135deg,#eef3f9,#ffffff);
      box-shadow:0 10px 28px rgba(44,74,124,.1); }
    .demo-tour-kicker { color:var(--accent); font-size:.76rem; font-weight:800; letter-spacing:.12em; }
    .demo-tour h2 { color:var(--text-dark); margin:.25rem 0 .35rem; }
    .demo-tour p { color:var(--text-muted); margin:0; line-height:1.5; }
    .demo-boundary { padding:.85rem 1rem; border:1px solid #d9a441; border-radius:8px;
      background:#fff8ea; color:#765516; font-size:.86rem; line-height:1.5; margin:.85rem 0 1rem; }
    @media (max-width: 800px) {
      .workflow { grid-template-columns:1fr; }
      .workflow-step { border-right:0; border-bottom:1px solid var(--border); }
      .importance-row { grid-template-columns:1fr; gap:.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
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
        st.session_state["chat_payment_id"] = payment_id


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
    shortcut_columns = st.columns(5)
    with shortcut_columns[0]:
        st.button(
            "Inspect risk cases",
            key="demo_open_transactions",
            use_container_width=True,
            on_click=_open_demo_view,
            args=("Transactions",),
        )
    with shortcut_columns[1]:
        st.button(
            "Open evidence report",
            key="demo_open_report",
            type="primary",
            use_container_width=True,
            disabled=report_payment_id is None,
            on_click=_open_demo_view,
            args=("Transactions", report_payment_id),
        )
    with shortcut_columns[2]:
        st.button(
            "Open policy audit",
            key="demo_open_insights",
            use_container_width=True,
            on_click=_open_demo_view,
            args=("Model insights",),
        )
    with shortcut_columns[3]:
        st.button(
            "Ask AI about a payment",
            key="demo_open_chat",
            use_container_width=True,
            on_click=_open_demo_view,
            args=("Ask about a payment", report_payment_id),
        )
    with shortcut_columns[4]:
        st.button(
            "Score a transaction",
            key="demo_open_manual_score",
            use_container_width=True,
            on_click=_open_demo_view,
            args=("Score a transaction",),
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


def render_evidence_report(filtered: pd.DataFrame) -> None:
    report_candidates = filtered[filtered["risk_score"].ge(DEMO_BLOCKING_THRESHOLD)]
    if report_candidates.empty:
        return
    st.markdown(
        """
        <div class="report-spotlight">
          <div class="report-kicker">KEY DEMO MOMENT</div>
          <h2>Go beyond the risk score</h2>
          <p>The score starts the review; the evidence report explains the decision in plain language
          and shows the exact synthetic signals behind it. This is an AI-written explanation, not a
          new model decision.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    payment_id = st.selectbox(
        "High-risk demo transaction",
        report_candidates["payment_id"].astype(str).tolist(),
        key="demo_report_payment_id",
    )
    generate = st.button(
        "Generate full evidence report",
        type="primary",
        use_container_width=True,
    )
    if generate:
        row = report_candidates.loc[
            report_candidates["payment_id"].astype(str).eq(payment_id)
        ].iloc[0]
        transaction = {
            "payment_id": str(row["payment_id"]), "velocity": int(row["velocity"]),
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


def render_transactions_view(transactions: pd.DataFrame, *, is_mock: bool) -> None:
    render_page_header(
        "Risk review transactions" if is_mock else "Account transactions",
        "Generated demonstration transactions with per-row risk signals."
        if is_mock else "Payment history loaded from the connected Razorpay Test Mode account.",
    )
    if is_mock:
        st.info("Demo signals are simulated; they are not real Razorpay fraud findings.")
    one, two, three = st.columns(3)
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
    if is_mock and not filtered.empty:
        render_evidence_report(filtered)


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


def render_model_insights(transactions: pd.DataFrame) -> None:
    render_page_header("Model insights")
    st.markdown("### Held-out performance")
    evaluation_path = PROJECT_ROOT / "backend" / "reports" / "metrics" / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
    metrics = evaluation.get("metrics", evaluation)
    try:
        curve = load_threshold_curve(THRESHOLD_CURVE_PATH)
        priced = cost_curve_for_ratio(curve, 5, 500)
        default_policy = cheapest_threshold_row(priced)
    except (FileNotFoundError, ValueError):
        curve = None
        default_policy = {"precision": 0.1335, "recall": 0.7661, "total_cost": 41835}
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Precision", f"{default_policy['precision']:.1%}")
    h2.metric("Recall", f"{default_policy['recall']:.1%}")
    h3.metric("PR-AUC", f"{float(metrics.get('average_precision', 0.2982)):.3f}")
    h4.metric("Expected cost", f"${default_policy['total_cost']:,.0f}")
    st.info(
        "Low precision by design: this policy accepts more false alarms to catch more fraud. "
        "Use the cost controls below to inspect the trade-off."
    )
    st.markdown("### Cost-of-fraud policy explorer")
    if curve is None:
        st.info("Held-out threshold curve unavailable.")
    else:
        left, right = st.columns(2)
        with left:
            cost_fp = st.slider("False-positive cost ($)", 1, 200, 5)
        with right:
            cost_fn = st.slider("False-negative cost ($)", 10, 2000, 500, 10)
        priced = cost_curve_for_ratio(curve, cost_fp, cost_fn)
        cheapest = cheapest_threshold_row(priced)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Cheapest threshold", f"{cheapest['threshold']:.2f}")
        p2.metric("Precision there", f"{cheapest['precision']:.1%}")
        p3.metric("Recall there", f"{cheapest['recall']:.1%}")
        p4.metric("Total held-out cost", f"${cheapest['total_cost']:,.0f}")
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
    st.markdown(
        '<div class="defense-banner">Defense-only: these views explain and prioritize existing signals '
        'for human review. They never automatically capture or refund a payment.</div>',
        unsafe_allow_html=True,
    )


def render_chat_view(transactions: pd.DataFrame) -> None:
    render_page_header("Ask about a payment", "Ask questions using only the selected transaction's fields.")
    if transactions.empty:
        st.info("No transaction is available for chat.")
        return
    payment_id = st.selectbox(
        "Transaction", transactions["payment_id"].astype(str).tolist(), key="chat_payment_id"
    )
    row = transactions.loc[transactions["payment_id"].astype(str).eq(payment_id)].iloc[0]
    st.markdown(
        f"**{row['status'].title()}** · {row['currency']} {row['amount']:,.2f} · "
        f"{row['method'].title()} · {row['created_at']:%d %b %Y, %H:%M UTC}"
    )
    chats = st.session_state.setdefault("razorpay_transaction_chats", {})
    history = chats.setdefault(payment_id, [])
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input(f"Ask about {payment_id}", key="razorpay_transaction_question")
    if not question:
        return
    prior = history[-8:]
    history.append({"role": "user", "content": question})
    context = {
        "payment_id": str(row["payment_id"]), "transaction_id": str(row["payment_id"]),
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


def render_manual_scoring_view() -> None:
    render_page_header(
        "Score a transaction",
        "Enter raw payment fields, run the trained model, then ask AI about its saved evidence.",
    )
    st.info(
        "Enter raw transaction data. The backend derives velocity, amount deviation, device history, "
        "geography mismatch, and recency from transactions scored earlier in this running session."
    )
    default_id = st.session_state.setdefault(
        "manual_transaction_id",
        f"manual_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
    )
    with st.form("manual_score_form"):
        left, middle, right = st.columns(3)
        with left:
            transaction_id = st.text_input("Transaction ID", value=default_id)
            timestamp = st.text_input(
                "Timestamp (UTC)", value=pd.Timestamp.now(tz="UTC").isoformat()
            )
            amount = st.number_input("Amount", min_value=0.0, value=1499.0, step=1.0)
        with middle:
            user_id = st.text_input("User ID", value="user_demo")
            device_id = st.text_input("Device ID", value="device_demo")
            card_id = st.text_input("Card ID", value="card_demo")
        with right:
            billing_country = st.text_input("Billing country", value="IN")
            ip_country = st.text_input("IP country", value="IN")
            merchant_category = st.text_input("Merchant category", value="electronics")
        submitted = st.form_submit_button(
            "Run fraud model", type="primary", use_container_width=True
        )

    if submitted:
        transaction = pd.DataFrame([{
            "transaction_id": transaction_id.strip(),
            "timestamp": timestamp.strip(),
            "user_id": user_id.strip(),
            "device_id": device_id.strip(),
            "card_id": card_id.strip(),
            "amount": float(amount),
            "billing_country": billing_country.strip().upper(),
            "ip_country": ip_country.strip().upper(),
            "merchant_category": merchant_category.strip(),
        }])
        try:
            scored = score_uploaded_transactions(transaction, SCORING_API_URL)
            st.session_state["manual_scored_transaction"] = scored.iloc[0].to_dict()
            st.session_state["manual_transaction_id"] = transaction_id.strip()
            st.session_state.setdefault("manual_transaction_chats", {}).setdefault(
                transaction_id.strip(), []
            )
        except (ScoringAPIError, ValueError) as exc:
            st.error(str(exc))

    result = st.session_state.get("manual_scored_transaction")
    if not result:
        st.caption("Start the backend, complete the form, and click Run fraud model.")
        return

    score = float(result["score"])
    flagged = bool(result["flagged"])
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Risk score", f"{score:.3f}")
    metric_right.metric(
        "Model result", "Flagged for review" if flagged else "Below review threshold"
    )
    reasons = list(result.get("reasons") or [])
    if reasons:
        st.markdown("#### Recorded model reasons")
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.caption("No elevated SHAP reason codes were recorded for this transaction.")
    st.warning(
        "This is an uncalibrated risk score for human review, not proof that the payment is fraudulent."
    )

    transaction_id = str(result["transaction_id"])
    st.markdown("### Ask about this scored transaction")
    history = st.session_state.setdefault("manual_transaction_chats", {}).setdefault(
        transaction_id, []
    )
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input(
        f"Ask why {transaction_id} received this score",
        key="manual_transaction_question",
    )
    if not question:
        return
    prior = history[-8:]
    history.append({"role": "user", "content": question})
    try:
        response = ask_transaction_question(
            transaction_id, question, prior, SCORING_API_URL
        )
        answer = (
            response["answer"]
            if response["status"] == "generated"
            else response.get("error", "Unable to answer.")
        )
    except ScoringAPIError as exc:
        answer = str(exc)
    history.append({"role": "assistant", "content": answer})
    st.rerun()


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
        brand_lockup(text_size="1.2rem", text_color="#16233D", stroke="#2C4A7C"),
        unsafe_allow_html=True,
    )
    navigation = [
        "Review queue",
        "Score a transaction",
        "Transactions",
        "Model insights",
        "Ask about a payment",
        "Razorpay account",
    ]
    if st.session_state.get("fraudlens_view") not in navigation:
        st.session_state["fraudlens_view"] = navigation[0]
    view = st.radio(
        "Navigation",
        navigation,
        index=0,
        key="fraudlens_view",
        format_func=lambda item: f"●  {item}",
    )
    st.markdown(
        '<div class="defense-footer">Defense-only. No model decision automatically changes a payment.</div>',
        unsafe_allow_html=True,
    )

top_left, top_right = st.columns([4, 1], vertical_alignment="center")
account_id = connection.get("razorpay_account_id", "")
connection_name = "Demo mode" if is_mock_session else "Razorpay connected"
with top_left:
    st.markdown(
        f'<div class="connection-bar"><span class="connection-led"></span>{connection_name} · '
        f'{"Synthetic transactions" if is_mock_session else account_id}</div>',
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
elif view == "Score a transaction":
    render_manual_scoring_view()
elif view == "Razorpay account":
    render_account_view(connection)
else:
    transactions = load_dashboard_transactions(show_date_filter=view == "Transactions")
    if transactions.empty:
        st.info("No transactions were found for the selected date range.")
    elif view == "Transactions":
        render_transactions_view(transactions, is_mock=is_mock_session)
    elif view == "Model insights":
        if is_mock_session:
            render_model_insights(transactions)
        else:
            render_page_header("Model insights")
            st.info("Model insights are available only for the synthetic dataset with required risk signals.")
    else:
        render_chat_view(transactions)
