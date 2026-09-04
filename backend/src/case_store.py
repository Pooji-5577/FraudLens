"""Durable case-management storage for the synthetic/demo fraud review workflow.

Case status and analyst notes are tracked separately from real Test Mode
enforcement (backend/src/review_store.py): per CONTEXT.md, simulated/demo
review must remain visibly distinct from real, money-moving enforcement.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from backend.src.review_store import SupabaseRESTClient, SupabaseStoreError

CASE_STATUSES = ("open", "under_investigation", "confirmed_fraud", "false_positive")


class CaseStore:
    """PostgREST-backed store for the fraud_cases and fraud_case_notes tables."""

    def __init__(
        self,
        url: str,
        key: str,
        *,
        http_client: Any | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.client = SupabaseRESTClient(url, key, http_client=http_client, timeout=timeout)

    @classmethod
    def from_environment(cls, *, http_client: Any | None = None) -> "CaseStore":
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

    def get_case(self, transaction_id: str) -> dict[str, Any] | None:
        rows = self.client.request(
            "GET",
            "fraud_cases",
            params={"transaction_id": f"eq.{transaction_id}", "select": "*", "limit": "1"},
        )
        return rows[0] if rows else None

    def list_cases(self, *, status: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"select": "*", "order": "updated_at.desc", "limit": "1000"}
        if status:
            params["status"] = f"eq.{status}"
        return self.client.request("GET", "fraud_cases", params=params)

    def set_status(
        self,
        transaction_id: str,
        status: str,
        *,
        actor: str = "",
        risk_score: float | None = None,
    ) -> dict[str, Any]:
        if status not in CASE_STATUSES:
            raise ValueError(f"status must be one of {CASE_STATUSES}")
        payload: dict[str, Any] = {
            "transaction_id": transaction_id,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": actor or None,
        }
        if risk_score is not None:
            payload["risk_score"] = risk_score
        rows = self.client.request(
            "POST",
            "fraud_cases",
            params={"on_conflict": "transaction_id"},
            payload=payload,
            prefer="resolution=merge-duplicates,return=representation",
        )
        return rows[0] if rows else payload

    def _ensure_case_exists(self, transaction_id: str) -> None:
        """Create an 'open' case row only if one doesn't already exist.

        Uses ignore-duplicates (not merge) so an existing case's status is
        never silently reset back to 'open' by adding a note to it.
        """
        self.client.request(
            "POST",
            "fraud_cases",
            params={"on_conflict": "transaction_id"},
            payload={
                "transaction_id": transaction_id,
                "status": "open",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            prefer="resolution=ignore-duplicates,return=minimal",
        )

    def add_note(self, transaction_id: str, note: str, *, author: str = "") -> dict[str, Any]:
        stripped = note.strip()
        if not stripped:
            raise ValueError("note must not be empty")
        self._ensure_case_exists(transaction_id)
        rows = self.client.request(
            "POST",
            "fraud_case_notes",
            payload={
                "transaction_id": transaction_id,
                "note": stripped,
                "author": author or None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            prefer="return=representation",
        )
        return rows[0] if rows else {}

    def list_notes(self, transaction_id: str) -> list[dict[str, Any]]:
        return self.client.request(
            "GET",
            "fraud_case_notes",
            params={"transaction_id": f"eq.{transaction_id}", "select": "*", "order": "id.asc"},
        )
