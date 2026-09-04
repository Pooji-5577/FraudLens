import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import time

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import requests

from frontend.processing import (
    ask_preview_transaction_question,
    ask_transaction_question,
    cheapest_threshold_row,
    chronological_transactions,
    cost_curve_for_ratio,
    csv_injection_safe,
    demo_case_catalog,
    filter_transactions_by_date,
    load_global_importance,
    load_threshold_curve,
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


def test_demo_case_catalog_surfaces_every_synthetic_case():
    transactions = pd.DataFrame([
        {"payment_id": "low", "risk_score": .20, "risk_status": "Low risk", "actual": "Legitimate"},
        {"payment_id": "review", "risk_score": .50, "risk_status": "Review", "actual": "Legitimate"},
        {"payment_id": "high", "risk_score": .90, "risk_status": "High risk", "actual": "Fraud"},
        {"payment_id": "fp", "risk_score": .80, "risk_status": "High risk", "actual": "Legitimate"},
        {"payment_id": "fn", "risk_score": .10, "risk_status": "Low risk", "actual": "Fraud"},
    ])

    catalog = demo_case_catalog(transactions, threshold=.65)

    assert catalog["Case"].tolist() == [
        "Low risk / allowed", "Review band", "High risk / report", "False positive", "False negative"
    ]
    assert catalog["Rows"].tolist() == [2, 1, 2, 1, 1]
    assert catalog["Example payment"].tolist() == ["low", "review", "high", "fp", "fn"]


def test_dashboard_http_client_round_trips_through_fastapi():
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    api_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
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
            cwd=root / "frontend",
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


def _threshold_curve():
    return pd.DataFrame([
        {"threshold": 0.1, "precision": 0.05, "recall": 0.95, "f1": 0.09, "false_positive_rate": 3610 / 14705, "tp": 190, "fp": 3610, "tn": 11095, "fn": 10},
        {"threshold": 0.5, "precision": 0.20, "recall": 0.60, "f1": 0.30, "false_positive_rate": 480 / 14705, "tp": 120, "fp": 480, "tn": 14225, "fn": 80},
        {"threshold": 0.9, "precision": 0.60, "recall": 0.10, "f1": 0.17, "false_positive_rate": 13 / 14705, "tp": 20, "fp": 13, "tn": 14692, "fn": 180},
    ])


def test_load_threshold_curve_reads_expected_columns(tmp_path):
    csv_path = tmp_path / "threshold_curve.csv"
    _threshold_curve().to_csv(csv_path, index=False)

    curve = load_threshold_curve(csv_path)

    assert list(curve["threshold"]) == [0.1, 0.5, 0.9]
    assert curve["fp"].sum() == 3610 + 480 + 13


def test_load_threshold_curve_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad_curve.csv"
    pd.DataFrame([{"threshold": 0.5, "precision": 0.2}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="threshold curve requires columns"):
        load_threshold_curve(csv_path)


def test_cost_curve_for_ratio_recomputes_total_cost_from_confusion_counts():
    priced = cost_curve_for_ratio(_threshold_curve(), cost_fp=5.0, cost_fn=500.0)

    assert list(priced["total_cost"]) == [
        3610 * 5.0 + 10 * 500.0,
        480 * 5.0 + 80 * 500.0,
        13 * 5.0 + 180 * 500.0,
    ]


def test_cost_curve_for_ratio_rejects_negative_costs():
    with pytest.raises(ValueError, match="non-negative"):
        cost_curve_for_ratio(_threshold_curve(), cost_fp=-1.0, cost_fn=500.0)


def test_csv_injection_safe_escapes_formula_prefixed_cells():
    frame = pd.DataFrame({
        "order_id": ["=cmd|'/c calc'!A1", "+1-800-555", "-2+3", "@SUM(A1)", "order_normal"],
        "email": ["safe@example.com", "\tattack", "\rattack", "plain", "another@example.com"],
        "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
    })

    safe = csv_injection_safe(frame)

    assert safe["order_id"].tolist() == [
        "'=cmd|'/c calc'!A1", "'+1-800-555", "'-2+3", "'@SUM(A1)", "order_normal",
    ]
    assert safe["email"].tolist() == ["safe@example.com", "'\tattack", "'\rattack", "plain", "another@example.com"]
    assert safe["amount"].tolist() == [100.0, 200.0, 300.0, 400.0, 500.0]


def test_csv_injection_safe_leaves_frame_without_risky_cells_unchanged():
    frame = pd.DataFrame({"payment_id": ["pay_1", "pay_2"], "status": ["captured", "failed"]})

    safe = csv_injection_safe(frame)

    pd.testing.assert_frame_equal(safe, frame)


def test_load_global_importance_sorts_signals_highest_first(tmp_path):
    json_path = tmp_path / "global_feature_importance.json"
    json_path.write_text(json.dumps({
        "held_out_rows": 15000,
        "signal_importance_percent": {"New device": 3.0, "Amount deviation": 42.4, "Time of day": 12.1},
    }))

    payload = load_global_importance(json_path)

    assert list(payload["signal_importance_percent"]) == ["Amount deviation", "Time of day", "New device"]


def test_load_global_importance_rejects_missing_signal_percentages(tmp_path):
    json_path = tmp_path / "bad.json"
    json_path.write_text(json.dumps({"held_out_rows": 15000}))

    with pytest.raises(ValueError, match="signal_importance_percent"):
        load_global_importance(json_path)


def test_cheapest_threshold_row_picks_lowest_total_cost_for_the_ratio():
    priced = cost_curve_for_ratio(_threshold_curve(), cost_fp=5.0, cost_fn=500.0)
    cheapest = cheapest_threshold_row(priced)
    assert cheapest["threshold"] == 0.1

    heavy_fp_priced = cost_curve_for_ratio(_threshold_curve(), cost_fp=500.0, cost_fn=500.0)
    cheapest_heavy_fp = cheapest_threshold_row(heavy_fp_priced)
    assert cheapest_heavy_fp["threshold"] == 0.9
