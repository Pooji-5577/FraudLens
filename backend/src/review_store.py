"""Durable review storage backends for FraudLens.

SQLite remains available for the isolated local/demo test path. Production-like
review state can be stored in the project's Supabase Data API using a
server-side secret key.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from backend.src.razorpay_enforcement import EnforcementError, ReviewStore


LOGGER = logging.getLogger(__name__)


class SupabaseStoreError(EnforcementError):
    """A database or Supabase Data API failure safe to show to an operator."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SupabaseRESTClient:
    """Small, dependency-free PostgREST client for the four review tables."""

    def __init__(
        self,
        url: str,
        key: str,
        *,
        timeout: float = 15.0,
        http_client: Any | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.http_client = http_client or requests.Session()
        if not self.url.startswith("https://"):
            raise SupabaseStoreError("SUPABASE_URL must be an https:// Supabase project URL.")
        if not self.key:
            raise SupabaseStoreError("A server-side Supabase secret key is required.")

    def request(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, Any] | None = None,
        payload: Any | None = None,
        prefer: str | None = None,
    ) -> list[dict[str, Any]]:
        headers = {"apikey": self.key, "Content-Type": "application/json"}
        if prefer:
            headers["Prefer"] = prefer
        try:
            response = self.http_client.request(
                method,
                f"{self.url}/rest/v1/{table}",
                params=params,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise SupabaseStoreError("Supabase timed out while accessing review storage.") from exc
        except requests.ConnectionError as exc:
            raise SupabaseStoreError("Supabase review storage is unreachable.") from exc
        except requests.RequestException as exc:
            raise SupabaseStoreError("Supabase rejected the review-storage request.") from exc

        if not 200 <= response.status_code < 300:
            detail = ""
            try:
                body = response.json()
                if isinstance(body, dict):
                    detail = str(body.get("message") or body.get("hint") or body.get("error") or "")
            except ValueError:
                pass
            suffix = f" Details: {detail}" if detail else ""
            raise SupabaseStoreError(
                f"Supabase review storage returned HTTP {response.status_code}.{suffix}",
                status_code=response.status_code,
            )

        if response.status_code == 204 or not response.content:
            return []
        try:
            body = response.json()
        except ValueError as exc:
            raise SupabaseStoreError("Supabase returned an invalid review-storage response.") from exc
        if body is None:
            return []
        if not isinstance(body, list):
            raise SupabaseStoreError("Supabase returned an unexpected review-storage response.")
        return body


class SupabaseReviewStore:
    """PostgREST-backed equivalent of :class:`ReviewStore`.

    The secret key is only read by the server process. Public/browser roles are
    denied by the migration's RLS and grants, so this class is not suitable for
    browser-side use.
    """

    def __init__(
        self,
        url: str,
        key: str,
        *,
        http_client: Any | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.client = SupabaseRESTClient(
            url,
            key,
            http_client=http_client,
            timeout=timeout,
        )

    @classmethod
    def from_environment(cls, *, http_client: Any | None = None) -> "SupabaseReviewStore":
        url = os.getenv("SUPABASE_URL", "").strip()
        key = (
            os.getenv("SUPABASE_SECRET_KEY", "").strip()
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        )
        if not url:
            raise SupabaseStoreError("SUPABASE_URL is not configured.")
        if not key:
            raise SupabaseStoreError(
                "SUPABASE_SECRET_KEY is not configured. Use a server-side secret key, not the publishable key."
            )
        return cls(url, key, http_client=http_client)

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

    @staticmethod
    def _review_row(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        evidence = result.pop("evidence_json", None)
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except json.JSONDecodeError:
                evidence = None
        result["evidence"] = evidence
        return result

    def _select_one(self, table: str, params: dict[str, Any]) -> dict[str, Any] | None:
        rows = self.client.request("GET", table, params={**params, "select": "*", "limit": "1"})
        return rows[0] if rows else None

    def _delete_event(self, event_id: str) -> None:
        self.client.request(
            "DELETE",
            "processed_webhook_events",
            params={"event_id": f"eq.{event_id}"},
            prefer="return=minimal",
        )

    def healthcheck(self) -> None:
        """Verify both the Supabase project and the review schema are reachable."""
        self.client.request(
            "GET",
            "processed_webhook_events",
            params={"select": "event_id", "limit": "1"},
        )

    def process_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply one verified webhook with database-enforced event uniqueness."""
        event_type = str(payload.get("event") or "")
        event_payload = payload.get("payload") or {}
        payment = event_payload.get("payment", {}).get("entity", {})
        payment_id = str(payment.get("id") or "") or None

        # Validate before marking an event as processed. This prevents malformed
        # requests from poisoning the idempotency table without a SQL transaction.
        account_id = None
        if event_type == "account.app.authorization_revoked":
            account_id = self._account_id_from_revocation(payload)
            if not account_id:
                raise ValueError("account.app.authorization_revoked webhook is missing account.id")
        elif event_type in {"payment.authorized", "payment.captured"} and not payment_id:
            raise ValueError(f"{event_type} webhook is missing payment.id")

        inserted = self.client.request(
            "POST",
            "processed_webhook_events",
            payload={
                "event_id": event_id,
                "event_type": event_type,
                "payment_id": payment_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
            prefer="resolution=ignore-duplicates,return=representation",
        )
        if not inserted:
            return {"status": "duplicate", "duplicate": True}

        try:
            if event_type == "account.app.authorization_revoked":
                self.client.request(
                    "POST",
                    "authorization_revocations",
                    params={"on_conflict": "account_id"},
                    payload={
                        "account_id": account_id,
                        "event_id": event_id,
                        "revoked_at": datetime.now(timezone.utc).isoformat(),
                    },
                    prefer="resolution=merge-duplicates,return=minimal",
                )
                return {
                    "status": "authorization_revoked",
                    "duplicate": False,
                    "account_id": account_id,
                }

            if event_type not in {"payment.authorized", "payment.captured"}:
                return {"status": "ignored", "duplicate": False}

            already_captured = (
                event_type == "payment.captured"
                or bool(payment.get("captured"))
                or payment.get("status") == "captured"
            )
            existing = self._select_one(
                "payment_reviews",
                {"payment_id": f"eq.{payment_id}"},
            )
            if existing:
                if existing["review_status"] == "pending_review" and already_captured:
                    LOGGER.warning(
                        "Razorpay payment %s was captured before review; it cannot remain on hold.",
                        payment_id,
                    )
                    updated = self.client.request(
                        "PATCH",
                        "payment_reviews",
                        params={
                            "payment_id": f"eq.{payment_id}",
                            "review_status": "eq.pending_review",
                        },
                        payload={
                            "review_status": "already_captured",
                            "payment_status": "captured",
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                        prefer="return=representation",
                    )
                    return {
                        "status": "already_captured" if updated else str(existing["review_status"]),
                        "duplicate": False,
                    }
                return {"status": str(existing["review_status"]), "duplicate": False}

            review_status = "already_captured" if already_captured else "pending_review"
            if already_captured:
                LOGGER.warning(
                    "Razorpay payment %s is already captured; it cannot be held for review.", payment_id
                )
            inserted_review = self.client.request(
                "POST",
                "payment_reviews",
                payload={
                    "payment_id": payment_id,
                    "order_id": payment.get("order_id"),
                    "amount": int(payment.get("amount", 0)),
                    "currency": str(payment.get("currency") or ""),
                    "payment_status": "captured" if already_captured else str(payment.get("status") or "authorized"),
                    "review_status": review_status,
                    "fulfillment_status": "on_hold",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                prefer="resolution=ignore-duplicates,return=representation",
            )
            if not inserted_review:
                current = self._select_one("payment_reviews", {"payment_id": f"eq.{payment_id}"})
                return {
                    "status": str(current["review_status"]) if current else review_status,
                    "duplicate": False,
                }
            return {"status": review_status, "duplicate": False}
        except Exception:
            # The event marker was inserted by this request. Remove it so a
            # transient downstream failure can be retried safely.
            try:
                self._delete_event(event_id)
            except SupabaseStoreError:
                pass
            raise

    def is_authorization_revoked(self, account_id: str) -> bool:
        return self._select_one(
            "authorization_revocations",
            {"account_id": f"eq.{account_id}"},
        ) is not None

    def get_review(self, payment_id: str) -> dict[str, Any] | None:
        row = self._select_one("payment_reviews", {"payment_id": f"eq.{payment_id}"})
        return self._review_row(row) if row else None

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
        rows = self.client.request(
            "PATCH",
            "payment_reviews",
            params={"payment_id": f"eq.{payment_id}"},
            payload={
                "review_status": review_status,
                "payment_status": payment_status,
                "fulfillment_status": fulfillment_status,
                "updated_at": now,
                "decided_at": now,
                "decided_by": actor,
                "decision": decision,
            },
            prefer="return=representation",
        )
        if not rows:
            raise KeyError(payment_id)
        return self._review_row(rows[0])

    def mark_already_captured(self, payment_id: str) -> dict[str, Any]:
        rows = self.client.request(
            "PATCH",
            "payment_reviews",
            params={
                "payment_id": f"eq.{payment_id}",
                "review_status": "eq.pending_review",
            },
            payload={
                "review_status": "already_captured",
                "payment_status": "captured",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            prefer="return=representation",
        )
        if not rows:
            raise EnforcementError("Payment is no longer pending review.")
        return self._review_row(rows[0])

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
        self.client.request(
            "POST",
            "enforcement_audit_log",
            payload={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payment_id": payment_id,
                "action": action,
                "actor": actor,
                "outcome": outcome,
                "risk_score": review.get("risk_score"),
                "evidence_json": review.get("evidence"),
                "detail": detail,
            },
            prefer="return=minimal",
        )

    def list_audit_entries(self, payment_id: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"order": "id.asc", "limit": "1000"}
        if payment_id is not None:
            params["payment_id"] = f"eq.{payment_id}"
        return self.client.request("GET", "enforcement_audit_log", params=params)

    def list_reviews(self) -> list[dict[str, Any]]:
        rows = self.client.request(
            "GET",
            "payment_reviews",
            params={"order": "created_at.desc", "limit": "1000"},
        )
        return [self._review_row(row) for row in rows]


def review_store_from_environment(sqlite_path: str | Path) -> ReviewStore | SupabaseReviewStore:
    """Select the configured backend without making SQLite tests remote."""
    backend = os.getenv("FRAUDLENS_STORAGE", "sqlite").strip().lower()
    if backend == "sqlite":
        return ReviewStore(sqlite_path)
    if backend == "supabase":
        return SupabaseReviewStore.from_environment()
    raise SupabaseStoreError(
        "FRAUDLENS_STORAGE must be either 'sqlite' or 'supabase'."
    )
