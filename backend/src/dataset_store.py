"""Server-side Supabase storage for uploaded transaction datasets."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import PurePath
from typing import Any
from uuid import uuid4

import pandas as pd

from backend.src.review_store import SupabaseRESTClient, SupabaseStoreError


class DatasetStore:
    """Persist parsed spreadsheet rows and their real model outputs."""

    def __init__(
        self,
        url: str,
        key: str,
        *,
        http_client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.client = SupabaseRESTClient(url, key, http_client=http_client, timeout=timeout)

    @classmethod
    def from_environment(cls, *, http_client: Any | None = None) -> "DatasetStore":
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
    def _safe_filename(filename: str) -> str:
        normalized = filename.replace("\\", "/")
        return PurePath(normalized).name[:255] or "transactions.csv"

    @staticmethod
    def _rows(dataset_id: str, scored: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(scored.to_dict(orient="records"), start=1):
            timestamp = pd.Timestamp(row["timestamp"])
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "row_number": row_number,
                    "transaction_id": str(row["transaction_id"]),
                    "transaction_timestamp": timestamp.isoformat(),
                    "user_id": str(row["user_id"]),
                    "device_id": str(row["device_id"]),
                    "card_id": str(row["card_id"]),
                    "amount": float(row["amount"]),
                    "billing_country": str(row["billing_country"]),
                    "ip_country": str(row["ip_country"]),
                    "ip_address": None if pd.isna(row.get("ip_address")) else str(row.get("ip_address", "")),
                    "merchant_category": str(row["merchant_category"]),
                    "uploaded_velocity_per_hour": None
                    if pd.isna(row.get("uploaded_velocity_per_hour"))
                    else float(row.get("uploaded_velocity_per_hour")),
                    "score": float(row["score"]),
                    "flagged": bool(row["flagged"]),
                    "blocked": bool(row["blocked"]),
                    "reasons": list(row["reasons"]),
                }
            )
        return rows

    def save_scored_dataset(self, filename: str, scored: pd.DataFrame) -> str:
        if scored.empty:
            raise ValueError("A dataset must contain at least one scored transaction.")
        dataset_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        safe_filename = self._safe_filename(filename)
        self.client.request(
            "POST",
            "fraud_datasets",
            payload={
                "id": dataset_id,
                "filename": safe_filename,
                "row_count": len(scored),
                "status": "processing",
                "created_at": now,
            },
            prefer="return=minimal",
        )
        try:
            rows = self._rows(dataset_id, scored)
            for start in range(0, len(rows), 500):
                self.client.request(
                    "POST",
                    "fraud_dataset_rows",
                    payload=rows[start : start + 500],
                    prefer="return=minimal",
                )
            self.client.request(
                "PATCH",
                "fraud_datasets",
                params={"id": f"eq.{dataset_id}"},
                payload={"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat()},
                prefer="return=minimal",
            )
        except SupabaseStoreError:
            try:
                self.client.request(
                    "PATCH",
                    "fraud_datasets",
                    params={"id": f"eq.{dataset_id}"},
                    payload={
                        "status": "failed",
                        "error_message": "Dataset rows could not be persisted.",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    },
                    prefer="return=minimal",
                )
            except SupabaseStoreError:
                pass
            raise
        return dataset_id

    def healthcheck(self) -> None:
        self.client.request(
            "GET",
            "fraud_datasets",
            params={"select": "id", "limit": "1"},
        )
