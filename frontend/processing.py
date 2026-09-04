"""Pure helpers for safe dashboard batch processing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
from urllib.parse import quote


class ScoringAPIError(RuntimeError):
    """A safe, user-displayable scoring service failure."""


_FORMULA_TRIGGER_CHARS = ("=", "+", "-", "@", "\t", "\r")


def csv_injection_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefix string cells spreadsheet apps would interpret as formulas.

    Razorpay payment fields like order_id, email, and contact are external
    input; a cell starting with =, +, -, or @ can execute as a formula when
    the exported CSV is later opened in Excel or Sheets. A leading apostrophe
    is the standard escape spreadsheet apps honor to force text.
    """

    def escape(value: object) -> object:
        if isinstance(value, str) and value.startswith(_FORMULA_TRIGGER_CHARS):
            return "'" + value
        return value

    return frame.map(escape)


def load_threshold_curve(path: Path | str) -> pd.DataFrame:
    """Load the held-out (never-trained-on) per-threshold confusion counts."""
    curve = pd.read_csv(path)
    required = {"threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"threshold curve requires columns: {', '.join(sorted(missing))}")
    return curve


def cost_curve_for_ratio(curve: pd.DataFrame, cost_fp: float, cost_fn: float) -> pd.DataFrame:
    """Recompute total held-out cost at every threshold for a chosen cost-per-error ratio."""
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError("costs must be non-negative")
    priced = curve.copy()
    priced["total_cost"] = priced["fp"] * cost_fp + priced["fn"] * cost_fn
    return priced


def cheapest_threshold_row(priced_curve: pd.DataFrame) -> pd.Series:
    """Return the held-out row with the lowest total cost for the current price ratio."""
    if priced_curve.empty:
        raise ValueError("threshold curve is empty")
    return priced_curve.loc[priced_curve["total_cost"].idxmin()]


def load_global_importance(path: Path | str) -> dict:
    """Load the real held-out mean(|SHAP|) signal importance, sorted highest first."""
    payload = json.loads(Path(path).read_text())
    percentages = payload.get("signal_importance_percent")
    if not isinstance(percentages, dict) or not percentages:
        raise ValueError("global importance artifact is missing signal_importance_percent")
    payload["signal_importance_percent"] = dict(
        sorted(percentages.items(), key=lambda item: item[1], reverse=True)
    )
    return payload


def risk_evidence_summary(transactions: pd.DataFrame, threshold: float) -> dict:
    """Calculate labelled-batch outcomes and amount-weighted error costs."""
    required = {"payment_id", "amount", "currency", "risk_score", "risk_status", "actual"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"risk evidence requires columns: {', '.join(sorted(missing))}")
    labelled = transactions.copy()
    labelled["blocked"] = labelled["risk_score"].ge(threshold)
    labelled["is_fraud"] = labelled["actual"].eq("Fraud")
    true_positive = labelled["blocked"] & labelled["is_fraud"]
    false_positive = labelled["blocked"] & ~labelled["is_fraud"]
    false_negative = ~labelled["blocked"] & labelled["is_fraud"]
    actual_positive = labelled["is_fraud"]
    precision_denominator = int(labelled["blocked"].sum())
    recall_denominator = int(actual_positive.sum())
    return {
        "transactions": len(labelled),
        "blocked": precision_denominator,
        "correctly_caught": int(true_positive.sum()),
        "precision": float(true_positive.sum() / precision_denominator) if precision_denominator else 0.0,
        "recall": float(true_positive.sum() / recall_denominator) if recall_denominator else 0.0,
        "false_positive_costs": {
            str(currency): float(amount)
            for currency, amount in labelled.loc[false_positive].groupby("currency")["amount"].sum().items()
        },
        "false_negative_costs": {
            str(currency): float(amount)
            for currency, amount in labelled.loc[false_negative].groupby("currency")["amount"].sum().items()
        },
        "false_positive_ids": labelled.loc[false_positive, "payment_id"].astype(str).tolist(),
        "false_negative_ids": labelled.loc[false_negative, "payment_id"].astype(str).tolist(),
    }


def risk_audit_rows(transactions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Build reviewer-facing decision records directly from scored rows."""
    rows = []
    for _, transaction in transactions.sort_values("risk_score", ascending=False).iterrows():
        blocked = float(transaction["risk_score"]) >= threshold
        actual_fraud = transaction["actual"] == "Fraud"
        if blocked and actual_fraud:
            outcome = "Correctly blocked"
        elif blocked:
            outcome = "False positive"
        elif actual_fraud:
            outcome = "False negative"
        else:
            outcome = "Correctly allowed"
        reasons = []
        if float(transaction.get("velocity", 0)) > 1:
            reasons.append(f"{int(transaction['velocity'])} recent transactions")
        if bool(transaction.get("ip_billing_mismatch", False)):
            reasons.append("IP/billing mismatch")
        if bool(transaction.get("new_device", False)):
            reasons.append("New device")
        deviation = float(transaction.get("amount_deviation", 0))
        if abs(deviation) >= 50:
            reasons.append(f"Amount deviation {deviation:+.0f}%")
        timestamp = pd.Timestamp(transaction["created_at"])
        if timestamp.hour < 6 or timestamp.hour >= 23:
            reasons.append("Odd-hour transaction")
        rows.append({
            "Payment": str(transaction["payment_id"]),
            "Decision": "Blocked" if blocked else "Allowed",
            "Risk score": float(transaction["risk_score"]),
            "Outcome": outcome,
            "Evidence": " • ".join(reasons) or (
                f"Score {'met' if blocked else 'stayed below'} the {threshold:.2f} threshold"
            ),
        })
    return pd.DataFrame(rows)


def demo_case_catalog(transactions: pd.DataFrame, threshold: float = .65) -> pd.DataFrame:
    """Summarize every labelled outcome available in the synthetic demo data.

    The catalog is intentionally derived from the rows shown in the dashboard, so the
    walkthrough never claims that a case exists without an example payment behind it.
    It is only for synthetic/mock sessions; real Razorpay rows do not have the labels or
    model signals required to classify these cases.
    """
    required = {"payment_id", "risk_score", "risk_status", "actual"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"demo case catalog requires columns: {', '.join(sorted(missing))}")

    columns = ["Case", "Rows", "Example payment", "Example score", "What it demonstrates"]
    if transactions.empty:
        return pd.DataFrame(columns=columns)

    labelled = transactions.copy()
    labelled["_score"] = pd.to_numeric(labelled["risk_score"], errors="coerce")
    labelled = labelled.loc[labelled["_score"].notna()].copy()
    labelled["_blocked"] = labelled["_score"].ge(threshold)
    labelled["_fraud"] = labelled["actual"].eq("Fraud")
    labelled["_outcome"] = "Correctly allowed"
    labelled.loc[labelled["_blocked"] & labelled["_fraud"], "_outcome"] = "Correctly blocked"
    labelled.loc[labelled["_blocked"] & ~labelled["_fraud"], "_outcome"] = "False positive"
    labelled.loc[~labelled["_blocked"] & labelled["_fraud"], "_outcome"] = "False negative"

    definitions = [
        (
            "Low risk / allowed",
            labelled["risk_status"].eq("Low risk"),
            "Below the review band; the synthetic policy leaves it allowed.",
        ),
        (
            "Review band",
            labelled["risk_status"].eq("Review"),
            "Uncertain score band; a reviewer can inspect the supporting signals.",
        ),
        (
            "High risk / report",
            labelled["risk_status"].eq("High risk"),
            "At the demo block threshold; open the evidence report walkthrough.",
        ),
        (
            "False positive",
            labelled["_outcome"].eq("False positive"),
            "A synthetic legitimate label above the threshold; inspect customer-friction cost.",
        ),
        (
            "False negative",
            labelled["_outcome"].eq("False negative"),
            "A synthetic fraud label below the threshold; inspect missed-fraud cost.",
        ),
    ]
    catalog = []
    for case, mask, explanation in definitions:
        matching = labelled.loc[mask].sort_values("_score", ascending=False)
        sample = matching.iloc[0] if not matching.empty else None
        catalog.append(
            {
                "Case": case,
                "Rows": int(len(matching)),
                "Example payment": str(sample["payment_id"]) if sample is not None else "—",
                "Example score": float(sample["_score"]) if sample is not None else None,
                "What it demonstrates": explanation,
            }
        )
    return pd.DataFrame(catalog, columns=columns)


def filter_transactions_by_date(
    transactions: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    """Return transactions within an inclusive calendar-date range."""
    timestamps = pd.to_datetime(transactions["timestamp"], utc=True, errors="raise")
    start = pd.Timestamp(start_date, tz="UTC")
    end_exclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    return transactions.loc[timestamps.ge(start) & timestamps.lt(end_exclusive)].copy()


def chronological_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Normalize an uploaded batch into deterministic chronological order."""
    ordered = transactions.copy()
    if "timestamp" not in ordered.columns:
        raise ValueError("uploaded CSV must contain a timestamp column")
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="raise")
    tie_breakers = ["timestamp"]
    if "transaction_id" in ordered.columns:
        tie_breakers.append("transaction_id")
    return ordered.sort_values(tie_breakers, kind="mergesort", ignore_index=True)


def scoring_api_available(api_url: str, timeout: float = 1.5) -> bool:
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
        return response.ok and response.json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def score_uploaded_transactions(
    transactions: pd.DataFrame,
    api_url: str,
    timeout: float = 120.0,
    http_client=requests,
) -> pd.DataFrame:
    """Sort a batch, then score it exclusively through the HTTP API."""
    ordered = chronological_transactions(transactions)
    payload = ordered.copy()
    payload["timestamp"] = payload["timestamp"].map(lambda value: value.isoformat())
    try:
        response = http_client.post(
            f"{api_url.rstrip('/')}/score/batch",
            json=payload.to_dict(orient="records"),
            timeout=timeout,
        )
        response.raise_for_status()
        results = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("The scoring service timed out. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                pass
        suffix = f" Details: {detail}" if detail else ""
        raise ScoringAPIError(f"The scoring service rejected this batch.{suffix}") from exc
    except ValueError as exc:
        raise ScoringAPIError("The scoring service returned an invalid response.") from exc

    required = {"score", "flagged", "blocked", "reasons"}
    if not isinstance(results, list) or len(results) != len(ordered):
        raise ScoringAPIError("The scoring service returned an incomplete batch.")
    if any(not isinstance(row, dict) or not required <= row.keys() for row in results):
        raise ScoringAPIError("The scoring service returned an unexpected result format.")
    result = ordered.copy()
    for column in ("score", "flagged", "blocked", "reasons"):
        result[column] = [row[column] for row in results]
    return result


def generate_transaction_report(
    transaction_id: str,
    api_url: str,
    timeout: float = 30.0,
    http_client=requests,
) -> dict:
    """Request a report for context retained by the scoring API."""
    try:
        response = http_client.post(
            f"{api_url.rstrip('/')}/report/{transaction_id}",
            timeout=timeout,
        )
        response.raise_for_status()
        report = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("Report generation timed out. The score and reasons are still available.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                pass
        raise ScoringAPIError(detail or "The evidence report could not be generated.") from exc
    if not isinstance(report, dict) or "status" not in report or "evidence" not in report:
        raise ScoringAPIError("The scoring service returned an invalid report.")
    return report


def generate_demo_transaction_report(
    transaction: dict,
    api_url: str,
    threshold: float = .65,
    timeout: float = 30.0,
    http_client=requests,
) -> dict:
    """Generate a report from the exact signals in a synthetic dashboard row."""
    try:
        response = http_client.post(
            f"{api_url.rstrip('/')}/demo-report",
            json={"transaction": transaction, "threshold": threshold},
            timeout=timeout,
        )
        response.raise_for_status()
        report = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("Evidence report generation timed out. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                pass
        raise ScoringAPIError(detail or "The demo evidence report could not be generated.") from exc
    except ValueError as exc:
        raise ScoringAPIError("The scoring service returned an invalid report.") from exc
    if not isinstance(report, dict) or "status" not in report or "evidence" not in report:
        raise ScoringAPIError("The scoring service returned an invalid report.")
    return report


def ask_transaction_question(
    transaction_id: str,
    question: str,
    history: list[dict],
    api_url: str,
    timeout: float = 30.0,
    http_client=requests,
) -> dict:
    """Ask a grounded question about context retained by the scoring API."""
    encoded_id = quote(str(transaction_id), safe="")
    try:
        response = http_client.post(
            f"{api_url.rstrip('/')}/chat/{encoded_id}",
            json={"question": question, "history": history[-8:]},
            timeout=timeout,
        )
        response.raise_for_status()
        answer = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("Transaction chat timed out. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                pass
        raise ScoringAPIError(detail or "The transaction question could not be answered.") from exc
    except ValueError as exc:
        raise ScoringAPIError("The scoring service returned an invalid chat response.") from exc
    if not isinstance(answer, dict) or "status" not in answer or "transaction_id" not in answer:
        raise ScoringAPIError("The scoring service returned an invalid chat response.")
    return answer


def ask_preview_transaction_question(
    transaction: dict,
    question: str,
    history: list[dict],
    api_url: str,
    timeout: float = 30.0,
    http_client=requests,
) -> dict:
    """Ask about raw fields for a visible transaction before it is scored."""
    try:
        response = http_client.post(
            f"{api_url.rstrip('/')}/preview-chat",
            json={"transaction": transaction, "question": question, "history": history[-8:]},
            timeout=timeout,
        )
        response.raise_for_status()
        answer = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("Transaction chat timed out. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                pass
        raise ScoringAPIError(detail or "The transaction question could not be answered.") from exc
    except ValueError as exc:
        raise ScoringAPIError("The scoring service returned an invalid chat response.") from exc
    if not isinstance(answer, dict) or "status" not in answer or "transaction_id" not in answer:
        raise ScoringAPIError("The scoring service returned an invalid chat response.")
    return answer
