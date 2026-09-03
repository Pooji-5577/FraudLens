"""Razorpay-connected transaction dashboard."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from dashboard.processing import (
    ScoringAPIError,
    ask_preview_transaction_question,
    risk_audit_rows,
    risk_evidence_summary,
)
from dashboard.razorpay_oauth import (
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
        f'<div style="display:flex;align-items:center;gap:.5rem;">'
        f'<div style="width:{size};height:{size};flex:none;">{mark}</div>'
        f'<span style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;'
        f'font-size:{text_size};color:{text_color};letter-spacing:-.01em;">'
        f'FraudLens<span style="color:#8ec5ff;">.</span></span></div>'
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
    .login-shell { max-width: 560px; margin: 8vh auto 1.5rem; padding: 2.4rem 2.5rem;
      border: 1px solid var(--border); border-radius: 22px; background: white;
      box-shadow: 0 18px 48px rgba(12,36,81,.14); text-align: center; }
    .login-mark { width: 56px; height: 56px; border-radius: 16px; margin: 0 auto 1.15rem;
      display: grid; place-items: center; color: white; font-family: 'Space Grotesk', sans-serif;
      font-weight: 700; font-size: 1.5rem;
      background: linear-gradient(135deg, var(--navy), var(--blue)); }
    .login-shell h1 { color: var(--text-dark); font-size: 1.9rem; margin: 0 0 .65rem; }
    .login-shell p { color: var(--text-muted); line-height: 1.55; margin: 0; }
    .login-trust { display: flex; align-items: center; justify-content: center; gap: .4rem;
      color: var(--text-muted); font-size: .82rem; margin-top: 1rem; }
    .workflow { display:grid; grid-template-columns:repeat(3,1fr); margin:0 0 1.4rem;
      border:1px solid var(--border); border-radius:4px; overflow:hidden; background:white; }
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
    @media (max-width: 800px) {
      .workflow { grid-template-columns:1fr; }
      .workflow-step { border-right:0; border-bottom:1px solid var(--border); }
      .importance-row { grid-template-columns:1fr; gap:.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RAZORPAY_AUTH_DISABLED = os.getenv("RAZORPAY_AUTH_DISABLED", "false").lower() == "true"
RAZORPAY_CLIENT_ID = os.getenv("RAZORPAY_CLIENT_ID", "")
RAZORPAY_CLIENT_SECRET = os.getenv("RAZORPAY_CLIENT_SECRET", "")
RAZORPAY_REDIRECT_URI = os.getenv("RAZORPAY_REDIRECT_URI", "")
RAZORPAY_MODE = os.getenv("RAZORPAY_MODE", "test").lower()
# When Partner credentials are absent, default to the clearly labelled mock
# session for local demo safety. Set false to require real configuration.
RAZORPAY_MOCK_AUTH = os.getenv("RAZORPAY_MOCK_AUTH", "true").lower() == "true"
SCORING_API_URL = os.getenv("SCORING_API_URL", "http://localhost:8000").rstrip("/")
DEMO_BLOCKING_THRESHOLD = float(os.getenv("DEMO_BLOCKING_THRESHOLD", "0.65"))
DEMO_REVIEW_THRESHOLD = float(os.getenv("DEMO_REVIEW_THRESHOLD", "0.40"))


def oauth_is_configured() -> bool:
    values = (RAZORPAY_CLIENT_ID, RAZORPAY_CLIENT_SECRET, RAZORPAY_REDIRECT_URI)
    return all(value and not value.startswith("replace-with-") for value in values)


def render_connection_screen() -> None:
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
        f"""
        <div class="login-shell">
          <div class="login-mark">R</div>
          <h1>Connect your Razorpay account</h1>
          <p>Sign in securely with Razorpay to view and filter your account transactions. This app requests read-only access and never sees your Razorpay password.</p>
          <div class="login-trust">🔒 Secured by Razorpay OAuth — you approve access on Razorpay's own site</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if oauth_is_configured():
        st.link_button(
            "Connect Razorpay account",
            build_authorization_url(RAZORPAY_CLIENT_ID, RAZORPAY_REDIRECT_URI, issue_oauth_state()),
            type="primary",
            use_container_width=True,
        )
        st.caption("You will be redirected to Razorpay to approve read-only account access.")
    elif RAZORPAY_MOCK_AUTH:
        if st.button("Connect mock Razorpay account", type="primary", use_container_width=True):
            st.session_state["razorpay_connection"] = {
                "access_token": "mock-access-token",
                "razorpay_account_id": "acc_mock_demo",
                "mock": True,
            }
            st.rerun()
        st.info("Demo mode uses generated transactions and does not connect to or modify a real Razorpay account.")
    else:
        st.warning(
            "Razorpay connection is not configured yet. Add RAZORPAY_CLIENT_ID, "
            "RAZORPAY_CLIENT_SECRET, and RAZORPAY_REDIRECT_URI to the server environment."
        )
        st.button("Connect Razorpay account", disabled=True, use_container_width=True)
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


if not RAZORPAY_AUTH_DISABLED and "razorpay_connection" not in st.session_state:
    render_connection_screen()

with st.sidebar:
    st.markdown(
        f'<div style="margin-bottom:1.1rem;">{brand_lockup()}</div>',
        unsafe_allow_html=True,
    )
    if RAZORPAY_AUTH_DISABLED:
        st.warning("Authentication bypass is enabled for local testing.")
    else:
        account_id = st.session_state["razorpay_connection"].get("razorpay_account_id", "")
        if st.session_state["razorpay_connection"].get("mock"):
            st.success("Mock Razorpay account connected")
            st.caption("Demo transactions only")
        else:
            st.success("Razorpay account connected")
        if account_id:
            st.caption(f"Account: {account_id}")
        if st.button("Disconnect account", use_container_width=True):
            connection = st.session_state["razorpay_connection"]
            if not connection.get("mock"):
                try:
                    revoke_access_token(
                        connection["access_token"],
                        client_id=RAZORPAY_CLIENT_ID,
                        client_secret=RAZORPAY_CLIENT_SECRET,
                    )
                except RazorpayOAuthError as exc:
                    st.error(f"Account is still connected: {exc}")
                    st.stop()
            st.session_state.pop("razorpay_connection", None)
            st.session_state.pop("razorpay_payments", None)
            st.session_state.pop("razorpay_loaded_range", None)
            st.session_state.pop("razorpay_transaction_chats", None)
            st.rerun()

is_mock_session = bool(st.session_state["razorpay_connection"].get("mock"))
hero_kicker = "SYNTHETIC RISK DEMO" if is_mock_session else "RAZORPAY PAYMENT HISTORY"
hero_title = "Synthetic scoring walkthrough" if is_mock_session else "Razorpay payment history"
hero_copy = (
    "Explore generated payments with simulated risk signals, decisions, labels, and evaluation evidence."
    if is_mock_session
    else "Browse and filter payments from the connected Razorpay account. This view is for account review and does not score real payments."
)
st.markdown(
    f"""
    <div class="hero">
    <div style="margin-bottom:1rem;">{brand_lockup(size="30px", text_size="1.2rem", text_color="white", stroke="#8ec5ff")}</div>
    <div class="hero-kicker">{hero_kicker}</div>
    <h1>{hero_title}</h1>
    <p>{hero_copy}</p></div>
    """,
    unsafe_allow_html=True,
)
if not is_mock_session:
    st.info(
        "Real Razorpay payments don't yet include the signals (device history, IP geography, velocity) "
        "the trained model needs, so real payments are shown for review but not scored by the fraud "
        "model. Scoring runs on the synthetic dataset that includes these signals."
    )

workflow = (
    ("Load synthetic transactions", "Inspect simulated risk signals", "Review demo decisions")
    if is_mock_session
    else ("Load payment history", "Filter account activity", "Inspect payment details")
)
st.markdown(
    f"""
    <div class="workflow">
      <div class="workflow-step"><div class="workflow-number">01</div><div class="workflow-title">{workflow[0]}</div></div>
      <div class="workflow-step"><div class="workflow-number">02</div><div class="workflow-title">{workflow[1]}</div></div>
      <div class="workflow-step"><div class="workflow-number">03</div><div class="workflow-title">{workflow[2]}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if RAZORPAY_AUTH_DISABLED and "razorpay_connection" not in st.session_state:
    st.info("Connect a real Razorpay account to load transactions. The authentication bypass only exposes the empty dashboard for local tests.")
    st.stop()

today = pd.Timestamp.now(tz="UTC").date()
default_start = today - pd.Timedelta(days=29)
with st.form("razorpay_date_filters"):
    date_column, action_column = st.columns([3, 1], vertical_alignment="bottom")
    with date_column:
        selected_dates = st.date_input(
            "Transaction date range (UTC)",
            value=(default_start, today),
            max_value=today,
        )
    with action_column:
        apply_dates = st.form_submit_button("Apply date range", type="primary", use_container_width=True)

should_load = apply_dates or "razorpay_payments" not in st.session_state
if should_load:
    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.warning("Choose both a start date and an end date.")
    else:
        range_start, range_end = selected_dates
        if range_start > range_end:
            st.warning("The start date must be before the end date.")
        else:
            from_timestamp = int(pd.Timestamp(range_start, tz="UTC").timestamp())
            to_timestamp = int(
                (pd.Timestamp(range_end, tz="UTC") + pd.Timedelta(days=1, seconds=-1)).timestamp()
            )
            try:
                with st.spinner("Loading transactions from Razorpay…"):
                    connection = st.session_state["razorpay_connection"]
                    if connection.get("mock"):
                        st.session_state["razorpay_payments"] = mock_payments(
                            from_timestamp, to_timestamp
                        )
                    else:
                        st.session_state["razorpay_payments"] = fetch_payments(
                            connection["access_token"],
                            from_timestamp=from_timestamp,
                            to_timestamp=to_timestamp,
                        )
                    st.session_state["razorpay_loaded_range"] = (range_start, range_end)
            except RazorpayOAuthError as exc:
                st.error(str(exc))

transactions = payments_frame(st.session_state.get("razorpay_payments", []))
if is_mock_session:
    st.warning(
        "Demo mode: these are generated examples with simulated risk signals and demonstration scores, "
        "not real Razorpay payments. The trained model is evaluated separately on the synthetic training dataset."
    )
if transactions.empty:
    st.info("No transactions were found for the selected date range.")
    st.stop()

st.markdown("### Filter transactions")
filter_one, filter_two, filter_three = st.columns(3)
with filter_one:
    search = st.text_input("Search", placeholder="Payment, order, email, or contact")
    selected_statuses = st.multiselect(
        "Status", sorted(value for value in transactions["status"].dropna().unique() if value)
    )
    selected_risk_statuses = []
    if transactions["risk_status"].notna().any():
        selected_risk_statuses = st.multiselect(
            "Risk status",
            [value for value in ("High risk", "Review", "Low risk") if value in set(transactions["risk_status"])],
        )
with filter_two:
    selected_methods = st.multiselect(
        "Payment method", sorted(value for value in transactions["method"].dropna().unique() if value)
    )
    selected_currencies = st.multiselect(
        "Currency", sorted(value for value in transactions["currency"].dropna().unique() if value)
    )
with filter_three:
    amount_min = float(transactions["amount"].min())
    amount_max = float(transactions["amount"].max())
    if amount_min == amount_max:
        st.number_input("Minimum amount", value=amount_min, disabled=True)
        st.number_input("Maximum amount", value=amount_max, disabled=True)
        selected_amounts = (amount_min, amount_max)
    else:
        selected_amounts = st.slider(
            "Amount range", min_value=amount_min, max_value=amount_max, value=(amount_min, amount_max)
        )
    international_only = st.checkbox("International payments only")

filtered = transactions.copy()
if search:
    needle = search.casefold()
    searchable = filtered[["payment_id", "order_id", "email", "contact"]].astype(str)
    filtered = filtered[searchable.apply(lambda column: column.str.casefold().str.contains(needle)).any(axis=1)]
if selected_statuses:
    filtered = filtered[filtered["status"].isin(selected_statuses)]
if selected_risk_statuses:
    filtered = filtered[filtered["risk_status"].isin(selected_risk_statuses)]
if selected_methods:
    filtered = filtered[filtered["method"].isin(selected_methods)]
if selected_currencies:
    filtered = filtered[filtered["currency"].isin(selected_currencies)]
filtered = filtered[filtered["amount"].between(*selected_amounts)]
if international_only:
    filtered = filtered[filtered["international"]]

captured = filtered[filtered["status"].eq("captured")]
failed = filtered[filtered["status"].eq("failed")]
metric_one, metric_two, metric_three, metric_four = st.columns(4)
metric_one.metric("Transactions", f"{len(filtered):,}")
metric_two.metric("Captured", f"{len(captured):,}")
metric_three.metric("Failed", f"{len(failed):,}")
metric_four.metric("Captured value", f"{captured['amount'].sum():,.2f}")

loaded_range = st.session_state.get("razorpay_loaded_range")
if loaded_range:
    st.caption(
        f"Showing {len(filtered):,} of {len(transactions):,} transactions loaded for "
        f"{loaded_range[0]:%d %b %Y} – {loaded_range[1]:%d %b %Y} (UTC)."
    )

is_risk_enriched = is_mock_session and filtered["risk_score"].notna().any()
if is_risk_enriched:
    st.markdown("### Risk review transactions")
    st.caption("Demo signals are simulated to demonstrate the review experience; they are not real Razorpay fraud findings.")
    displayed = pd.DataFrame(
        {
            "Txn": filtered["payment_id"],
            "Amount": filtered.apply(lambda row: f"{row['currency']} {row['amount']:,.2f}", axis=1),
            "Velocity": filtered["velocity"].map(lambda value: f"{int(value)} recent"),
            "IP/billing": filtered["ip_billing_mismatch"].map({True: "Mismatch", False: "Match"}),
            "Device": filtered["new_device"].map({True: "New", False: "Known"}),
            "Amt. dev.": filtered["amount_deviation"].map(lambda value: f"{float(value):+.0f}%"),
            "Hour": filtered["created_at"].dt.strftime("%H:%M"),
            "Score": filtered["risk_score"].astype(float),
            "Risk status": filtered["risk_status"],
            "Payment status": filtered["status"].str.title(),
            "Actual": filtered["actual"],
        }
    )
else:
    st.markdown("### Account transactions")
    st.info(
        "Razorpay's Payments API supplies payment details but not device history, IP/billing comparison, "
        "velocity, or ground-truth fraud labels. Those risk columns appear only when an enrichment source is connected."
    )
    displayed = filtered[
        ["payment_id", "created_at", "amount", "currency", "status", "method", "order_id", "email", "contact"]
    ]
st.dataframe(
    displayed,
    use_container_width=True,
    hide_index=True,
    height=560,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Risk score", min_value=0.0, max_value=1.0, format="%.3f"
        ),
        "payment_id": "Payment",
        "created_at": "Date (UTC)",
        "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
        "currency": "Currency",
        "status": "Status",
        "method": "Method",
        "order_id": "Order",
        "email": "Email",
        "contact": "Contact",
    },
)

if is_risk_enriched:
    st.markdown(
        """
        <div class="importance-panel">
          <div class="importance-title">What influences the demo risk score</div>
          <div class="importance-row"><span>Transaction velocity</span><div class="importance-track"><div class="importance-fill" style="width:35%"></div></div><strong>35%</strong></div>
          <div class="importance-row"><span>IP/billing mismatch</span><div class="importance-track"><div class="importance-fill" style="width:20%"></div></div><strong>20%</strong></div>
          <div class="importance-row"><span>New device</span><div class="importance-track"><div class="importance-fill" style="width:33%"></div></div><strong>33%</strong></div>
          <div class="importance-row"><span>Amount deviation</span><div class="importance-track"><div class="importance-fill" style="width:4%"></div></div><strong>4%</strong></div>
          <div class="importance-row"><span>Odd-hour transaction</span><div class="importance-track"><div class="importance-fill" style="width:8%"></div></div><strong>8%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Risk evaluation evidence")
    st.caption("Synthetic labels are revealed here only to evaluate the mock risk decisions.")
    evidence = risk_evidence_summary(filtered, threshold=DEMO_BLOCKING_THRESHOLD)
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Test transactions", f"{evidence['transactions']:,}")
    e2.metric("Flagged as fraud", f"{evidence['blocked']:,}")
    e3.metric("Correctly caught", f"{evidence['correctly_caught']:,}")
    e4.metric("Precision", f"{evidence['precision']:.1%}")
    e5.metric("Recall", f"{evidence['recall']:.1%}")

    cost_one, cost_two = st.columns(2)
    cost_one.metric(
        "False-positive cost (legitimate payments blocked)",
        format_currency_costs(evidence["false_positive_costs"]),
    )
    cost_two.metric(
        "False-negative cost (fraud that slipped through)",
        format_currency_costs(evidence["false_negative_costs"]),
    )
    st.caption(
        "These amount-weighted costs show the trade-off between blocking legitimate revenue and allowing labelled fraud through."
    )

    st.markdown("#### Error breakdown")
    error_rows = []
    if evidence["false_positive_ids"]:
        error_rows.append({
            "Error": "False positive — legitimate payment blocked",
            "Transactions": ", ".join(evidence["false_positive_ids"]),
            "Amount": format_currency_costs(evidence["false_positive_costs"]),
        })
    if evidence["false_negative_ids"]:
        error_rows.append({
            "Error": "False negative — fraud that slipped through",
            "Transactions": ", ".join(evidence["false_negative_ids"]),
            "Amount": format_currency_costs(evidence["false_negative_costs"]),
        })
    if error_rows:
        st.dataframe(pd.DataFrame(error_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No false-positive or false-negative errors in the filtered transactions.")

    st.markdown("#### Audit trail")
    st.caption("Each entry is derived from the transaction’s saved score, decision threshold, label, and signals.")
    st.dataframe(
        risk_audit_rows(filtered, threshold=DEMO_BLOCKING_THRESHOLD),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk score": st.column_config.NumberColumn("Risk score", format="%.3f"),
            "Evidence": st.column_config.TextColumn("Evidence", width="large"),
        },
    )
st.download_button(
    "Download filtered transactions",
    displayed.to_csv(index=False),
    "razorpay_transactions.csv",
    mime="text/csv",
)

st.divider()
st.markdown("### Ask AI about a transaction")
st.caption("Select a visible payment and ask questions using only its Razorpay transaction fields.")
if filtered.empty:
    st.info("No filtered transaction is available for chat.")
else:
    chat_payment_id = st.selectbox(
        "Transaction for AI chat",
        filtered["payment_id"].astype(str).tolist(),
        key="chat_payment_id",
    )
    chat_row = filtered.loc[filtered["payment_id"].astype(str).eq(chat_payment_id)].iloc[0]
    st.markdown(
        f"**{chat_row['status'].title()}** · {chat_row['currency']} {chat_row['amount']:,.2f} · "
        f"{chat_row['method'].title()} · {chat_row['created_at']:%d %b %Y, %H:%M UTC}"
    )
    chats = st.session_state.setdefault("razorpay_transaction_chats", {})
    history = chats.setdefault(chat_payment_id, [])
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(f"Ask about {chat_payment_id}", key="razorpay_transaction_question")
    if question:
        prior_history = history[-8:]
        history.append({"role": "user", "content": question})
        transaction_context = {
            "payment_id": str(chat_row["payment_id"]),
            "transaction_id": str(chat_row["payment_id"]),
            "timestamp": chat_row["created_at"].isoformat(),
            "amount": float(chat_row["amount"]),
            "currency": str(chat_row["currency"]),
            "status": str(chat_row["status"]),
            "method": str(chat_row["method"]),
            "order_id": str(chat_row["order_id"]),
            "email": str(chat_row["email"]),
            "contact": str(chat_row["contact"]),
            "international": bool(chat_row["international"]),
            "velocity": None if pd.isna(chat_row["velocity"]) else int(chat_row["velocity"]),
            "ip_billing_mismatch": (
                None if pd.isna(chat_row["ip_billing_mismatch"]) else bool(chat_row["ip_billing_mismatch"])
            ),
            "new_device": None if pd.isna(chat_row["new_device"]) else bool(chat_row["new_device"]),
            "amount_deviation": (
                None if pd.isna(chat_row["amount_deviation"]) else float(chat_row["amount_deviation"])
            ),
            "risk_score": None if pd.isna(chat_row["risk_score"]) else float(chat_row["risk_score"]),
            "risk_status": None if pd.isna(chat_row["risk_status"]) else str(chat_row["risk_status"]),
            "actual": None if pd.isna(chat_row["actual"]) else str(chat_row["actual"]),
        }
        try:
            with st.spinner("Checking the selected transaction…"):
                response = ask_preview_transaction_question(
                    transaction_context,
                    question,
                    prior_history,
                    SCORING_API_URL,
                )
            answer = (
                response["answer"]
                if response["status"] == "generated"
                else response.get("error", "The AI answer could not be generated.")
            )
        except ScoringAPIError as exc:
            answer = str(exc)
        history.append({"role": "assistant", "content": answer})
        st.rerun()
