"""Test-mode-only Razorpay authorization review and enforcement."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests


LOGGER = logging.getLogger(__name__)


def verify_webhook_signature(raw_body: bytes, provided_signature: str | None, secret: str) -> bool:
    """Verify Razorpay's HMAC-SHA256 signature over the untouched body."""
    if not provided_signature or not secret:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided_signature)


class ReviewStore:
    """SQLite-backed review, webhook-deduplication, and audit storage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS processed_webhook_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payment_id TEXT,
                    processed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS authorization_revocations (
                    account_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    revoked_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS payment_reviews (
                    payment_id TEXT PRIMARY KEY,
                    order_id TEXT,
                    amount INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    payment_status TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    fulfillment_status TEXT NOT NULL DEFAULT 'on_hold',
                    risk_score REAL,
                    evidence_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    decided_at TEXT,
                    decided_by TEXT,
                    decision TEXT
                );
                CREATE TABLE IF NOT EXISTS enforcement_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    payment_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    risk_score REAL,
                    evidence_json TEXT,
                    detail TEXT
                );
                """
            )

    def process_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply one verified webhook exactly once."""
        event_type = str(payload.get("event") or "")
        payment = payload.get("payload", {}).get("payment", {}).get("entity", {})
        payment_id = str(payment.get("id") or "") or None
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM processed_webhook_events WHERE event_id = ?", (event_id,)
            ).fetchone():
                return {"status": "duplicate", "duplicate": True}
            connection.execute(
                "INSERT INTO processed_webhook_events(event_id,event_type,payment_id,processed_at) VALUES(?,?,?,?)",
                (event_id, event_type, payment_id, now),
            )
            if event_type == "account.app.authorization_revoked":
                account_id = self._account_id_from_revocation(payload)
                if not account_id:
                    raise ValueError(
                        "account.app.authorization_revoked webhook is missing account.id"
                    )
                connection.execute(
                    "INSERT OR REPLACE INTO authorization_revocations(account_id,event_id,revoked_at) VALUES(?,?,?)",
                    (account_id, event_id, now),
                )
                return {
                    "status": "authorization_revoked",
                    "duplicate": False,
                    "account_id": account_id,
                }
            if event_type not in {"payment.authorized", "payment.captured"}:
                return {"status": "ignored", "duplicate": False}
            if not payment_id:
                raise ValueError(f"{event_type} webhook is missing payment.id")
            already_captured = (
                event_type == "payment.captured"
                or bool(payment.get("captured"))
                or payment.get("status") == "captured"
            )
            existing = connection.execute(
                "SELECT review_status FROM payment_reviews WHERE payment_id = ?", (payment_id,)
            ).fetchone()
            if existing:
                if existing["review_status"] == "pending_review" and already_captured:
                    LOGGER.warning(
                        "Razorpay payment %s was captured before review; it cannot remain on hold.",
                        payment_id,
                    )
                    connection.execute(
                        """
                        UPDATE payment_reviews
                        SET review_status='already_captured', payment_status='captured', updated_at=?
                        WHERE payment_id=?
                        """,
                        (now, payment_id),
                    )
                    return {"status": "already_captured", "duplicate": False}
                return {"status": str(existing["review_status"]), "duplicate": False}
            review_status = "already_captured" if already_captured else "pending_review"
            if already_captured:
                LOGGER.warning(
                    "Razorpay payment %s is already captured; it cannot be held for review.", payment_id
                )
            connection.execute(
                """
                INSERT INTO payment_reviews(
                    payment_id,order_id,amount,currency,payment_status,review_status,
                    fulfillment_status,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,'on_hold',?,?)
                """,
                (
                    payment_id,
                    payment.get("order_id"),
                    int(payment.get("amount", 0)),
                    str(payment.get("currency") or ""),
                    "captured" if already_captured else str(payment.get("status") or "authorized"),
                    review_status,
                    now,
                    now,
                ),
            )
            return {"status": review_status, "duplicate": False}

    @staticmethod
    def _account_id_from_revocation(payload: dict[str, Any]) -> str | None:
        event_payload = payload.get("payload") or {}
        for resource_name in ("account", "app"):
            resource = event_payload.get(resource_name) or {}
            entity = resource.get("entity") if isinstance(resource, dict) else None
            if isinstance(entity, dict) and entity.get("id"):
                return str(entity["id"])
        for key in ("account_id", "razorpay_account_id"):
            if event_payload.get(key):
                return str(event_payload[key])
            if payload.get(key):
                return str(payload[key])
        return None

    def is_authorization_revoked(self, account_id: str) -> bool:
        """Return whether Razorpay has revoked this app's account authorization."""
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM authorization_revocations WHERE account_id = ?", (account_id,)
            ).fetchone() is not None

    def get_review(self, payment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM payment_reviews WHERE payment_id = ?", (payment_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["evidence"] = json.loads(result.pop("evidence_json") or "null")
        return result

    def complete_decision(
        self,
        payment_id: str,
        *,
        review_status: str,
        payment_status: str,
        fulfillment_status: str,
        actor: str,
        decision: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE payment_reviews
                SET review_status=?, payment_status=?, fulfillment_status=?, updated_at=?,
                    decided_at=?, decided_by=?, decision=?
                WHERE payment_id=?
                """,
                (
                    review_status,
                    payment_status,
                    fulfillment_status,
                    now,
                    now,
                    actor,
                    decision,
                    payment_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(payment_id)
        return self.get_review(payment_id)  # type: ignore[return-value]

    def mark_already_captured(self, payment_id: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE payment_reviews
                SET review_status='already_captured', payment_status='captured', updated_at=?
                WHERE payment_id=? AND review_status='pending_review'
                """,
                (now, payment_id),
            )
            if cursor.rowcount != 1:
                raise EnforcementError("Payment is no longer pending review.")
        return self.get_review(payment_id)  # type: ignore[return-value]

    def append_audit(
        self,
        payment_id: str,
        *,
        action: str,
        actor: str,
        outcome: str,
        detail: str | None = None,
    ) -> None:
        review = self.get_review(payment_id)
        if review is None:
            raise KeyError(payment_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO enforcement_audit_log(
                    timestamp,payment_id,action,actor,outcome,risk_score,evidence_json,detail
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    payment_id,
                    action,
                    actor,
                    outcome,
                    review.get("risk_score"),
                    json.dumps(review.get("evidence")) if review.get("evidence") is not None else None,
                    detail,
                ),
            )

    def list_audit_entries(self, payment_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM enforcement_audit_log"
        parameters: tuple[Any, ...] = ()
        if payment_id is not None:
            query += " WHERE payment_id = ?"
            parameters = (payment_id,)
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_reviews(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payment_id FROM payment_reviews ORDER BY created_at DESC"
            ).fetchall()
        return [self.get_review(row["payment_id"]) for row in rows]  # type: ignore[misc]


class EnforcementError(RuntimeError):
    """A safe enforcement failure that can be displayed to a reviewer."""


class RazorpayGateway:
    """Small bearer-authenticated Razorpay API boundary for review actions."""

    API_ROOT = "https://api.razorpay.com/v1"

    def __init__(self, access_token: str, *, timeout: float = 15.0, http_client=requests) -> None:
        if not access_token:
            raise EnforcementError("A server-side Razorpay access token is required.")
        self.headers = {"Authorization": f"Bearer {access_token}"}
        self.timeout = timeout
        self.http_client = http_client

    def _payload(self, response: Any) -> dict[str, Any]:
        try:
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise EnforcementError("Razorpay enforcement timed out; check payment state before retrying.") from exc
        except requests.RequestException as exc:
            raise EnforcementError("Razorpay rejected the enforcement request.") from exc
        except ValueError as exc:
            raise EnforcementError("Razorpay returned an invalid enforcement response.") from exc
        if not isinstance(payload, dict):
            raise EnforcementError("Razorpay returned an invalid enforcement response.")
        return payload

    def fetch_payment(self, payment_id: str) -> dict[str, Any]:
        response = self.http_client.get(
            f"{self.API_ROOT}/payments/{quote(payment_id, safe='')}",
            headers=self.headers,
            timeout=self.timeout,
        )
        return self._payload(response)

    def fetch_order(self, order_id: str) -> dict[str, Any]:
        response = self.http_client.get(
            f"{self.API_ROOT}/orders/{quote(order_id, safe='')}",
            headers=self.headers,
            timeout=self.timeout,
        )
        return self._payload(response)

    def capture_payment(self, payment_id: str, *, amount: int, currency: str) -> dict[str, Any]:
        response = self.http_client.post(
            f"{self.API_ROOT}/payments/{quote(payment_id, safe='')}/capture",
            headers=self.headers,
            json={"amount": int(amount), "currency": currency},
            timeout=self.timeout,
        )
        return self._payload(response)

    def refund_payment(self, payment_id: str, *, idempotency_key: str) -> dict[str, Any]:
        response = self.http_client.post(
            f"{self.API_ROOT}/payments/{quote(payment_id, safe='')}/refund",
            headers={**self.headers, "X-Refund-Idempotency": idempotency_key},
            json={},
            timeout=self.timeout,
        )
        return self._payload(response)


class EnforcementService:
    """Human-triggered, Test Mode-only payment decisions."""

    def __init__(self, store: ReviewStore, gateway: Any, *, mode: str) -> None:
        if mode.lower() != "test":
            raise EnforcementError("Razorpay enforcement is restricted to Test Mode.")
        self.store = store
        self.gateway = gateway

    @staticmethod
    def _actor(actor: str) -> str:
        actor = actor.strip()
        if not actor:
            raise EnforcementError("Reviewer identity is required.")
        return actor

    def approve_and_capture(self, payment_id: str, *, actor: str) -> dict[str, Any]:
        actor = self._actor(actor)
        try:
            return self._approve_and_capture(payment_id, actor)
        except EnforcementError as exc:
            if self.store.get_review(payment_id) is not None:
                self.store.append_audit(
                    payment_id,
                    action="capture",
                    actor=actor,
                    outcome="failed",
                    detail=str(exc),
                )
            raise

    def _approve_and_capture(self, payment_id: str, actor: str) -> dict[str, Any]:
        review = self.store.get_review(payment_id)
        if review is None:
            raise EnforcementError("Payment is not in the FraudLens review queue.")
        payment = self.gateway.fetch_payment(payment_id)
        if payment.get("status") == "captured" or payment.get("captured") is True:
            if review["review_status"] not in {"pending_review", "approved_captured"}:
                raise EnforcementError("This payment already has a different review decision.")
            if review["review_status"] != "approved_captured":
                review = self.store.complete_decision(
                    payment_id,
                    review_status="approved_captured",
                    payment_status="captured",
                    fulfillment_status="approved",
                    actor=actor,
                    decision="approve_and_capture",
                )
            self.store.append_audit(
                payment_id,
                action="capture",
                actor=actor,
                outcome="already_captured",
                detail="Razorpay already reports this payment as captured; no capture call was sent.",
            )
            return review
        if payment.get("status") != "authorized" or payment.get("captured") is True:
            raise EnforcementError("Only an authorized, uncaptured payment can be captured.")
        order_id = review.get("order_id")
        if not order_id:
            raise EnforcementError("The payment has no Razorpay order; exact capture cannot be verified.")
        order = self.gateway.fetch_order(order_id)
        amount = int(order.get("amount", -1))
        currency = str(order.get("currency") or "")
        if amount != review["amount"] or currency != review["currency"]:
            raise EnforcementError("Razorpay order amount or currency does not match the held payment.")
        self.gateway.capture_payment(payment_id, amount=amount, currency=currency)
        result = self.store.complete_decision(
            payment_id,
            review_status="approved_captured",
            payment_status="captured",
            fulfillment_status="approved",
            actor=actor,
            decision="approve_and_capture",
        )
        self.store.append_audit(
            payment_id,
            action="capture",
            actor=actor,
            outcome="succeeded",
            detail=f"Captured {amount} {currency} in Razorpay Test Mode.",
        )
        return result

    def confirm_fraud(self, payment_id: str, *, actor: str) -> dict[str, Any]:
        actor = self._actor(actor)
        try:
            return self._confirm_fraud(payment_id, actor)
        except EnforcementError as exc:
            if self.store.get_review(payment_id) is not None:
                self.store.append_audit(
                    payment_id,
                    action="confirm_fraud",
                    actor=actor,
                    outcome="failed",
                    detail=str(exc),
                )
            raise

    def _confirm_fraud(self, payment_id: str, actor: str) -> dict[str, Any]:
        review = self.store.get_review(payment_id)
        if review is None:
            raise EnforcementError("Payment is not in the FraudLens review queue.")
        if review["review_status"] == "confirmed_fraud":
            self.store.append_audit(
                payment_id,
                action="confirm_fraud",
                actor=actor,
                outcome="already_confirmed",
                detail="Fraud was already confirmed; no Razorpay call was sent.",
            )
            return review
        if review["review_status"] != "pending_review":
            raise EnforcementError("Only a pending authorization can be released as confirmed fraud.")
        payment = self.gateway.fetch_payment(payment_id)
        if payment.get("status") != "authorized" or payment.get("captured") is True:
            if payment.get("status") == "captured" or payment.get("captured") is True:
                self.store.mark_already_captured(payment_id)
            raise EnforcementError(
                "Razorpay no longer reports an uncaptured authorization; refresh and use the refund action if captured."
            )
        result = self.store.complete_decision(
            payment_id,
            review_status="confirmed_fraud",
            payment_status="authorized",
            fulfillment_status="stopped",
            actor=actor,
            decision="confirm_fraud_and_release_authorization",
        )
        self.store.append_audit(
            payment_id,
            action="confirm_fraud",
            actor=actor,
            outcome="succeeded",
            detail="Capture was withheld; fulfillment stopped in FraudLens.",
        )
        return result

    def refund_and_stop(self, payment_id: str, *, actor: str) -> dict[str, Any]:
        actor = self._actor(actor)
        try:
            return self._refund_and_stop(payment_id, actor)
        except EnforcementError as exc:
            if self.store.get_review(payment_id) is not None:
                self.store.append_audit(
                    payment_id,
                    action="refund",
                    actor=actor,
                    outcome="failed",
                    detail=str(exc),
                )
            raise

    def _refund_and_stop(self, payment_id: str, actor: str) -> dict[str, Any]:
        review = self.store.get_review(payment_id)
        if review is None:
            raise EnforcementError("Payment is not in the FraudLens review queue.")
        payment = self.gateway.fetch_payment(payment_id)
        fully_refunded = (
            payment.get("status") == "refunded"
            or int(payment.get("amount_refunded") or 0) >= int(payment.get("amount") or review["amount"])
        )
        if fully_refunded:
            if review["review_status"] not in {"already_captured", "refunded"}:
                raise EnforcementError("This payment already has a different review decision.")
            if review["review_status"] != "refunded":
                review = self.store.complete_decision(
                    payment_id,
                    review_status="refunded",
                    payment_status="refunded",
                    fulfillment_status="stopped",
                    actor=actor,
                    decision="refund_and_stop_fulfillment",
                )
            self.store.append_audit(
                payment_id,
                action="refund",
                actor=actor,
                outcome="already_refunded",
                detail="Razorpay already reports a full refund; no refund call was sent.",
            )
            return review
        if payment.get("status") != "captured":
            raise EnforcementError("Only a captured payment can be refunded.")
        if review["review_status"] not in {"already_captured", "refunded"}:
            raise EnforcementError("This payment is not eligible for the captured-payment refund action.")
        idempotency_key = "fraudlens-refund-" + hashlib.sha256(payment_id.encode()).hexdigest()[:24]
        self.gateway.refund_payment(payment_id, idempotency_key=idempotency_key)
        result = self.store.complete_decision(
            payment_id,
            review_status="refunded",
            payment_status="refunded",
            fulfillment_status="stopped",
            actor=actor,
            decision="refund_and_stop_fulfillment",
        )
        self.store.append_audit(
            payment_id,
            action="refund",
            actor=actor,
            outcome="succeeded",
            detail="Requested a full Razorpay Test Mode refund and stopped fulfillment.",
        )
        return result
