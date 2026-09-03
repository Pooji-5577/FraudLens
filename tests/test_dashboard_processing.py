import os
from pathlib import Path
import subprocess
import sys
import socket
import time

import pandas as pd
from pandas.testing import assert_frame_equal
import requests

from dashboard.processing import (
    ask_preview_transaction_question,
    ask_transaction_question,
    chronological_transactions,
    filter_transactions_by_date,
    risk_audit_rows,
    risk_evidence_summary,
    score_uploaded_transactions,
)


def _transactions():
    base = pd.Timestamp("2026-06-01T10:00:00Z")
    return pd.DataFrame([
        {
            "transaction_id": f"order-{i}",
            "timestamp": (base + pd.Timedelta(minutes=10 * i)).isoformat(),
            "user_id": "order-user",
            "device_id": "order-device",
            "card_id": "order-card",
            "amount": 100 + i,
            "billing_country": "IN",
            "ip_country": "IN",
            "merchant_category": "grocery",
        }
        for i in range(5)
    ])


class _Response:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        return None

    def json(self):
        return self.rows


class _RecordingHTTPClient:
    def __init__(self):
        self.payload = None

    def post(self, _url, json, timeout):
        self.payload = json
        rows = [
            {"score": row["amount"] / 1_000, "flagged": False, "blocked": False, "reasons": []}
            for row in json
        ]
        return _Response(rows)


def test_shuffled_upload_is_sent_to_api_in_chronological_order():
    ordered = chronological_transactions(_transactions())
    shuffled = _transactions().sample(frac=1, random_state=11).reset_index(drop=True)
    client = _RecordingHTTPClient()
    actual = score_uploaded_transactions(shuffled, "http://scoring.test", http_client=client)
    expected = ordered.assign(
        score=ordered["amount"] / 1_000,
        flagged=False,
        blocked=False,
        reasons=[[] for _ in range(len(ordered))],
    )
    assert_frame_equal(actual, expected)
    assert [row["transaction_id"] for row in client.payload] == ordered["transaction_id"].tolist()


def test_transaction_date_filter_is_inclusive():
    transactions = _transactions()
    transactions.loc[3:, "timestamp"] = [
        "2026-06-02T00:00:00Z",
        "2026-06-03T00:00:00Z",
    ]

    filtered = filter_transactions_by_date(transactions, "2026-06-01", "2026-06-02")

    assert filtered["transaction_id"].tolist() == ["order-0", "order-1", "order-2", "order-3"]


def test_risk_evidence_counts_errors_costs_and_audit_rows():
    transactions = pd.DataFrame([
        {"payment_id": "tp", "amount": 100.0, "currency": "INR", "risk_score": .8,
         "risk_status": "High risk", "actual": "Fraud", "velocity": 8,
         "ip_billing_mismatch": True, "new_device": False, "amount_deviation": 10,
         "created_at": "2026-06-01T12:00:00Z"},
        {"payment_id": "fp", "amount": 40.0, "currency": "INR", "risk_score": .7,
         "risk_status": "High risk", "actual": "Legitimate", "velocity": 2,
         "ip_billing_mismatch": False, "new_device": True, "amount_deviation": 70,
         "created_at": "2026-06-01T23:00:00Z"},
        {"payment_id": "fn", "amount": 25.0, "currency": "USD", "risk_score": .3,
         "risk_status": "Low risk", "actual": "Fraud", "velocity": 1,
         "ip_billing_mismatch": False, "new_device": False, "amount_deviation": 5,
         "created_at": "2026-06-01T12:00:00Z"},
        {"payment_id": "tn", "amount": 20.0, "currency": "INR", "risk_score": .1,
         "risk_status": "Low risk", "actual": "Legitimate", "velocity": 1,
         "ip_billing_mismatch": False, "new_device": False, "amount_deviation": 5,
         "created_at": "2026-06-01T12:00:00Z"},
    ])

    summary = risk_evidence_summary(transactions, threshold=.65)
    audit = risk_audit_rows(transactions, threshold=.65)

    assert summary["transactions"] == 4
    assert summary["blocked"] == 2
    assert summary["correctly_caught"] == 1
    assert summary["precision"] == .5
    assert summary["recall"] == .5
    assert summary["false_positive_costs"] == {"INR": 40.0}
    assert summary["false_negative_costs"] == {"USD": 25.0}
    assert summary["false_positive_ids"] == ["fp"]
    assert summary["false_negative_ids"] == ["fn"]
    assert set(audit["Outcome"]) == {
        "Correctly blocked", "False positive", "False negative", "Correctly allowed"
    }


def test_dashboard_http_client_round_trips_through_fastapi():
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    api_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                if requests.get(f"{api_url}/health", timeout=.25).ok:
                    break
            except requests.RequestException:
                time.sleep(.1)
        else:
            raise AssertionError("FastAPI test server did not start")

        transaction = _transactions().iloc[[0]].copy()
        result = score_uploaded_transactions(transaction, api_url, timeout=30)
        assert len(result) == 1
        assert isinstance(result.iloc[0]["score"], float)
        assert pd.api.types.is_bool_dtype(result["flagged"])
        assert pd.api.types.is_bool_dtype(result["blocked"])
        assert isinstance(result.iloc[0]["reasons"], list)
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_dashboard_script_resolves_project_packages_from_its_own_directory():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys, streamlit as st; "
                "st.stop = lambda: sys.exit(0); "
                "runpy.run_path('app.py', run_name='__main__')"
            ),
        ],
        cwd=root / "dashboard",
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_transaction_question_is_sent_with_id_and_bounded_history():
    class ChatClient:
        def __init__(self):
            self.url = None
            self.payload = None

        def post(self, url, json, timeout):
            self.url = url
            self.payload = json
            return _Response({
                "status": "generated",
                "answer": "The payment was blocked because its score exceeded the threshold.",
                "transaction_id": "txn/42",
            })

    client = ChatClient()
    history = [{"role": "user", "content": str(i)} for i in range(12)]

    result = ask_transaction_question(
        "txn/42", "Why was it blocked?", history, "http://scoring.test", http_client=client
    )

    assert client.url == "http://scoring.test/chat/txn%2F42"
    assert client.payload == {"question": "Why was it blocked?", "history": history[-8:]}
    assert result["status"] == "generated"


def test_preview_transaction_question_sends_visible_row_without_scoring():
    class ChatClient:
        def __init__(self):
            self.url = None
            self.payload = None

        def post(self, url, json, timeout):
            self.url = url
            self.payload = json
            return _Response({
                "status": "generated",
                "answer": "The amount is 100.00.",
                "transaction_id": "order-0",
            })

    client = ChatClient()
    transaction = _transactions().iloc[0].to_dict()

    result = ask_preview_transaction_question(
        transaction,
        "What is the amount?",
        [],
        "http://scoring.test",
        http_client=client,
    )

    assert client.url == "http://scoring.test/preview-chat"
    assert client.payload["transaction"] == transaction
    assert client.payload["question"] == "What is the amount?"
    assert result["answer"] == "The amount is 100.00."
