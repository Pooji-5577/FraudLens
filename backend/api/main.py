"""FastAPI wrapper around the stateful fraud scorer."""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.src.score import FraudScorer
from backend.src.case_store import CaseStore
from backend.src.razorpay_enforcement import ReviewStore, verify_webhook_signature
from backend.src.review_store import (
    SupabaseReviewStore,
    SupabaseStoreError,
    review_store_from_environment,
)
from backend.src.report import (
    answer_preview_transaction_question,
    answer_transaction_question,
    generate_demo_report,
    generate_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="Fraud Spike Detector", version="1.0.0")
scorer = FraudScorer()
report_contexts: dict[str, dict] = {}


class Transaction(BaseModel):
    transaction_id: str
    timestamp: datetime
    user_id: str
    device_id: str
    card_id: str
    amount: float = Field(ge=0)
    billing_country: str
    ip_country: str
    merchant_category: str


class ScoreResponse(BaseModel):
    score: float
    flagged: bool
    blocked: bool
    reasons: list[str]
    report: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class TransactionChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)


class PreviewTransactionChatRequest(TransactionChatRequest):
    transaction: dict[str, Any]


class DemoReportRequest(BaseModel):
    transaction: dict[str, Any]
    threshold: float = Field(default=.65, ge=0, le=1)


class CaseStatusRequest(BaseModel):
    status: str
    actor: str = ""
    risk_score: float | None = None


class CaseNoteRequest(BaseModel):
    note: str = Field(min_length=1, max_length=4_000)
    author: str = ""


def _case_store() -> CaseStore:
    try:
        return CaseStore.from_environment()
    except SupabaseStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/storage")
def storage_health() -> dict[str, str]:
    """Verify the configured review-storage backend and Supabase schema."""
    database_path = Path(
        os.getenv("RAZORPAY_ENFORCEMENT_DB", "backend/data/razorpay_enforcement.sqlite3")
    )
    try:
        store = review_store_from_environment(database_path)
        if isinstance(store, SupabaseReviewStore):
            store.healthcheck()
            return {"status": "ok", "backend": "supabase"}
        return {"status": "ok", "backend": "sqlite"}
    except SupabaseStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/webhooks/razorpay", status_code=202)
async def razorpay_webhook(request: Request) -> dict[str, Any]:
    """Authenticate a Razorpay webhook before parsing or processing it."""
    if os.getenv("RAZORPAY_MODE", "test").lower() != "test":
        raise HTTPException(status_code=503, detail="Razorpay enforcement is restricted to Test Mode.")
    raw_body = await request.body()
    if not verify_webhook_signature(
        raw_body,
        request.headers.get("x-razorpay-signature"),
        os.getenv("RAZORPAY_WEBHOOK_SECRET", ""),
    ):
        raise HTTPException(status_code=400, detail="Invalid Razorpay webhook signature.")
    event_id = request.headers.get("x-razorpay-event-id")
    if not event_id:
        raise HTTPException(status_code=400, detail="Missing Razorpay webhook event ID.")
    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid Razorpay webhook payload.") from exc
    database_path = Path(
        os.getenv("RAZORPAY_ENFORCEMENT_DB", "backend/data/razorpay_enforcement.sqlite3")
    )
    store = getattr(app.state, "razorpay_enforcement", None)
    storage_backend = os.getenv("FRAUDLENS_STORAGE", "sqlite").lower()
    if storage_backend == "supabase":
        if not isinstance(store, SupabaseReviewStore):
            store = review_store_from_environment(database_path)
    else:
        if (
            not isinstance(store, ReviewStore)
            or getattr(store, "path", None) != database_path
        ):
            store = review_store_from_environment(database_path)
    app.state.razorpay_enforcement = store
    try:
        return store.process_event(event_id, payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SupabaseStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
def score(transaction: Transaction, include_report: bool = False) -> dict:
    payload = transaction.model_dump()
    payload["timestamp"] = payload["timestamp"].isoformat()
    try:
        return scorer.score_one(payload, include_report=include_report)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(transactions: list[Transaction]) -> list[dict]:
    """Score one chronological batch while preserving point-in-time state."""
    if not transactions:
        return []
    payloads = []
    for transaction in transactions:
        payload = transaction.model_dump()
        payload["timestamp"] = payload["timestamp"].isoformat()
        payloads.append(payload)
    frame = pd.DataFrame(payloads)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values(["timestamp", "transaction_id"], kind="mergesort", ignore_index=True)
    try:
        scored = scorer.score_batch(frame)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for row in scored.to_dict(orient="records"):
        report_contexts[str(row["transaction_id"])] = row
    return scored[["score", "flagged", "blocked", "reasons"]].to_dict(orient="records")


@app.post("/report/{transaction_id}", response_model=dict[str, Any])
def report(transaction_id: str) -> dict:
    context = report_contexts.get(transaction_id)
    if context is None:
        raise HTTPException(status_code=404, detail="No scored transaction context was found for this ID.")
    generated = generate_report(
        context,
        float(context["score"]),
        bool(context["blocked"]),
        scorer.threshold,
        list(context["reasons"]),
    )
    if generated is None:
        raise HTTPException(
            status_code=503,
            detail="Narrative reports are unavailable because Azure OpenAI is not configured.",
        )
    return generated


@app.post("/demo-report", response_model=dict[str, Any])
def demo_report(request: DemoReportRequest) -> dict:
    try:
        generated = generate_demo_report(request.transaction, request.threshold)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="The synthetic demo signals are incomplete.") from exc
    if generated is None:
        raise HTTPException(
            status_code=503,
            detail="AI evidence reports are unavailable or the selected demo payment is below threshold.",
        )
    return generated


@app.post("/chat/{transaction_id}", response_model=dict[str, Any])
def chat_about_transaction(transaction_id: str, request: TransactionChatRequest) -> dict:
    context = report_contexts.get(transaction_id)
    if context is None:
        raise HTTPException(status_code=404, detail="No scored transaction context was found for this ID.")
    generated = answer_transaction_question(
        context,
        float(context["score"]),
        bool(context["blocked"]),
        scorer.threshold,
        list(context["reasons"]),
        request.question,
        [message.model_dump() for message in request.history],
    )
    if generated is None:
        raise HTTPException(
            status_code=503,
            detail="Transaction chat is unavailable because Azure OpenAI is not configured.",
        )
    return generated


@app.post("/preview-chat", response_model=dict[str, Any])
def chat_about_preview_transaction(request: PreviewTransactionChatRequest) -> dict:
    transaction = request.transaction
    generated = answer_preview_transaction_question(
        transaction,
        request.question,
        [message.model_dump() for message in request.history],
    )
    if generated is None:
        raise HTTPException(
            status_code=503,
            detail="Transaction chat is unavailable because Azure OpenAI is not configured.",
        )
    return generated


@app.get("/cases", response_model=list[dict[str, Any]])
def list_cases(status: str | None = None) -> list[dict]:
    try:
        return _case_store().list_cases(status=status)
    except SupabaseStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/cases/{transaction_id}", response_model=dict[str, Any])
def get_case(transaction_id: str) -> dict:
    try:
        store = _case_store()
        return {"case": store.get_case(transaction_id), "notes": store.list_notes(transaction_id)}
    except SupabaseStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/cases/{transaction_id}/status", response_model=dict[str, Any])
def set_case_status(transaction_id: str, request: CaseStatusRequest) -> dict:
    try:
        return _case_store().set_status(
            transaction_id, request.status, actor=request.actor, risk_score=request.risk_score
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SupabaseStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/cases/{transaction_id}/notes", response_model=dict[str, Any])
def add_case_note(transaction_id: str, request: CaseNoteRequest) -> dict:
    try:
        return _case_store().add_note(transaction_id, request.note, author=request.author)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SupabaseStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
