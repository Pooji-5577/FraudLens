"""Presentation-only payment enforcement state with no network dependencies."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


def initial_mock_scenarios() -> dict[str, dict[str, Any]]:
    """Return fresh Razorpay-shaped demo records for the two review outcomes."""
    return {
        "authorized": {
            "scenario": "Authorized — awaiting review",
            "payment_id": "pay_demo_0005",
            "order_id": "order_demo_0005",
            "amount": 149900,
            "currency": "INR",
            "method": "card",
            "billing_name": "Asha Rao",
            "billing_email": "asha.rao@example.com",
            "billing_contact": "+919876543210",
            "billing_country": "IN",
            "status": "Authorized — awaiting review",
            "fulfillment_status": "On hold",
        },
        "captured": {
            "scenario": "Captured before review",
            "payment_id": "pay_demo_0001",
            "order_id": "order_demo_0001",
            "amount": 89900,
            "currency": "INR",
            "method": "upi",
            "billing_name": "Kabir Shah",
            "billing_email": "kabir.shah@example.com",
            "billing_contact": "+919812345678",
            "billing_country": "IN",
            "status": "Captured before review",
            "fulfillment_status": "On hold",
        },
    }


def apply_mock_action(
    scenarios: dict[str, dict[str, Any]],
    audit_log: list[dict[str, str]],
    *,
    scenario_id: str,
    action: str,
    reviewer: str = "Demo Reviewer",
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Apply one local-only transition and append its immediately visible audit row."""
    scenario = scenarios[scenario_id]
    allowed = {
        ("authorized", "approve"): ("Approve & capture", "Captured", "Approved"),
        ("authorized", "confirm_fraud"): (
            "Confirm fraud & release authorization",
            "Capture withheld",
            "Stopped",
        ),
        ("captured", "refund"): ("Refund & stop fulfillment", "Refunded", "Stopped"),
    }
    try:
        action_label, resulting_status, fulfillment_status = allowed[(scenario_id, action)]
    except KeyError as exc:
        raise ValueError("This action is not available for the selected demo scenario.") from exc

    expected_status = (
        "Authorized — awaiting review" if scenario_id == "authorized" else "Captured before review"
    )
    if scenario["status"] != expected_status:
        return deepcopy(scenario)

    scenario["status"] = resulting_status
    scenario["fulfillment_status"] = fulfillment_status
    observed_at = timestamp or datetime.now(timezone.utc)
    audit_log.append(
        {
            "Timestamp (UTC)": observed_at.isoformat(),
            "Payment ID": scenario["payment_id"],
            "Reviewer": reviewer,
            "Action": action_label,
            "Resulting status": resulting_status,
        }
    )
    return deepcopy(scenario)
