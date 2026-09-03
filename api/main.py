"""FastAPI wrapper around the stateful fraud scorer."""

from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

from src.score import FraudScorer
from src.report import (
    answer_preview_transaction_question,
    answer_transaction_question,
    generate_report,
)

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
