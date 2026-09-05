"""Pure helpers for safe dashboard batch processing."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
import re

import pandas as pd
import requests
from urllib.parse import quote


class ScoringAPIError(RuntimeError):
    """A safe, user-displayable scoring service failure."""


_FORMULA_TRIGGER_CHARS = ("=", "+", "-", "@", "\t", "\r")

_UPLOAD_COLUMN_ALIASES = {
    "transaction_id": {"transaction_id", "transaction", "transactionid", "txn", "txn_id", "txnid", "payment_id", "paymentid"},
    "timestamp": {"timestamp", "transaction_time", "transactiontime", "created_at", "createdat", "date_time", "datetime", "time"},
    "user_id": {"user_id", "userid", "user", "customer_id", "customerid", "customer"},
    "device_id": {"device_id", "deviceid", "device", "device_name", "devicename"},
    "card_id": {"card_id", "cardid", "card", "card_number", "cardnumber", "payment_instrument", "paymentinstrument"},
    "amount": {"amount", "transaction_amount", "transactionamount", "value", "payment_amount", "paymentamount"},
    "billing_country": {"billing_country", "billingcountry", "billing_country_code", "billingcountrycode", "country"},
    "ip_country": {"ip_country", "ipcountry", "ip_country_code", "ipcountrycode", "location_country", "locationcountry"},
    "merchant_category": {"merchant_category", "merchantcategory", "merchant_category_code", "merchantcategorycode", "mcc", "method", "payment_method", "paymentmethod"},
    "uploaded_velocity_per_hour": {
        "velocity", "velocity_per_hour", "velocityperhour", "txn_velocity", "transaction_velocity",
        "velocity_hr", "velocity_h", "transactions_per_hour",
    },
    "ip_address": {"ip_address", "ipaddress", "ip", "client_ip", "clientip"},
    "ip_billing": {
        "ip_billing", "ipbilling", "ip_billing_status", "ipbillingstatus", "geography", "geo_status",
        "geo_mismatch", "ip_billing_mismatch",
    },
    "amount_deviation": {"amount_deviation", "amountdeviation", "amount_dev", "amt_dev", "amtdeviation", "amount_delta", "amount_zscore"},
    "hour": {"hour", "transaction_hour", "transactionhour", "txn_hour", "hour_utc"},
    "actual": {"actual", "actual_label", "label", "is_fraud", "fraud", "fraud_label", "ground_truth", "groundtruth"},
}


def _normalized_upload_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lstrip("\ufeff").strip().casefold()).strip("_")


def _parse_velocity_per_hour(value: object) -> float | None:
    text = str(value).strip().casefold()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    count = float(match.group(1))
    return count / 24.0 if "24" in text else count


def _parse_optional_number(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).strip().replace(",", ""))
    if not match:
        return None
    parsed = float(match.group(0))
    return parsed if isfinite(parsed) else None


def _parse_hour(value: object) -> int | None:
    """Parse 24-hour values as well as friendly values such as ``10:18 PM``."""
    if value is None or value is pd.NA:
        return None
    text = str(value).strip().casefold()
    if not text:
        return None
    match = re.search(r"(\d{1,2})(?::\d{1,2})?\s*(a\.?m\.?|p\.?m\.?)?", text)
    if not match:
        return None
    hour = int(match.group(1))
    meridiem = (match.group(2) or "").replace(".", "")
    if meridiem:
        if hour < 1 or hour > 12:
            return None
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
    return hour if 0 <= hour <= 23 else None


def _normalize_ip_billing(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return "Mismatch" if value else "Match"
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "n/a", "na", "unknown"}:
        return None
    normalized = re.sub(r"[^a-z]+", " ", text.casefold()).strip()
    if normalized in {"match", "matched", "same", "same country", "no mismatch", "false", "0"}:
        return "Match"
    if normalized in {"mismatch", "mismatched", "different", "different country", "true", "1"}:
        return "Mismatch"
    return text


def _ip_billing_mismatch(value: object) -> bool | None:
    """Infer a mismatch from either a status or a pair of country codes."""
    normalized = _normalize_ip_billing(value)
    if normalized is None:
        return None
    if normalized == "Match":
        return False
    if normalized == "Mismatch":
        return True
    codes = [
        code for code in re.findall(r"\b[A-Z]{2}\b", str(normalized).upper())
        if code not in {"IP", "ID"}
    ]
    return codes[0] != codes[-1] if len(codes) >= 2 else None


def _normalize_actual(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return "Fraud" if value else "Legitimate"
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "n/a", "na", "unknown"}:
        return None
    normalized = text.casefold()
    if normalized in {"1", "true", "yes", "fraud", "fraudulent", "positive"}:
        return "Fraud"
    if normalized in {"0", "false", "no", "legitimate", "legit", "genuine", "negative"}:
        return "Legitimate"
    return text


def _ip_country_from_address(value: object, billing_country: object) -> str:
    """Best-effort fallback when a CSV has IP address but no IP country column."""
    ip = str(value).strip()
    billing = str(billing_country).strip().upper() or "UNKNOWN"
    if not ip:
        return billing
    if ip.startswith(
        ("10.", "127.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
         "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
         "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.")
    ):
        return billing
    if billing == "IN" and ip.startswith(("49.36.", "103.21.", "106.51.")):
        return "IN"
    return "EXTERNAL"


def normalize_uploaded_transactions(frame: pd.DataFrame) -> dict:
    """Map common CSV headers and fill absent model inputs deterministically.

    The fraud model has a fixed input contract. Uploads may use friendlier labels,
    so this boundary recognizes aliases and supplies conservative placeholders for
    fields the source does not contain. The returned metadata lets the UI disclose
    every mapping and inference instead of silently pretending the source was complete.
    """
    if frame.empty:
        raise ValueError("The uploaded CSV has no rows to score.")

    source_by_normalized = {
        _normalized_upload_header(column): column for column in frame.columns
    }
    selected: dict[str, object] = {}
    mapped: dict[str, str] = {}
    for target, aliases in _UPLOAD_COLUMN_ALIASES.items():
        source = next(
            (source_by_normalized[alias] for alias in aliases if alias in source_by_normalized),
            None,
        )
        if source is not None:
            selected[target] = source
            mapped[str(source)] = target

    row_numbers = pd.Series(range(1, len(frame) + 1), index=frame.index)
    output = pd.DataFrame(index=frame.index)
    inferred: list[str] = []

    if "transaction_id" in selected:
        output["transaction_id"] = frame[selected["transaction_id"]].astype(str).str.strip()
    else:
        output["transaction_id"] = row_numbers.map(lambda value: f"upload-row-{value:06d}")
        inferred.append("transaction_id")

    if "timestamp" in selected:
        output["timestamp"] = pd.to_datetime(frame[selected["timestamp"]], utc=True, errors="raise")
    elif "hour" in selected:
        parsed_hours = frame[selected["hour"]].map(_parse_hour)
        output["timestamp"] = pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(
            parsed_hours.fillna(0), unit="h"
        ) + pd.to_timedelta(row_numbers - 1, unit="s")
        inferred.append("timestamp")
    else:
        output["timestamp"] = pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(
            row_numbers - 1, unit="s"
        )
        inferred.append("timestamp")

    currency = pd.Series("INR", index=frame.index, dtype="object")
    if "amount" in selected:
        raw_amount = frame[selected["amount"]]
        detected_currency = raw_amount.astype(str).str.extract(r"\b([A-Za-z]{3})\b", expand=False)
        currency = detected_currency.str.upper().fillna("INR")
        cleaned_amount = raw_amount.astype(str).str.replace(",", "", regex=False).str.replace(
            r"[^0-9.\-]", "", regex=True
        )
        output["amount"] = pd.to_numeric(cleaned_amount, errors="raise")
    else:
        output["amount"] = 0.0
        inferred.append("amount")

    if "user_id" in selected:
        output["user_id"] = frame[selected["user_id"]].astype(str).str.strip()
    else:
        output["user_id"] = row_numbers.map(lambda value: f"inferred-user-{value:06d}")
        inferred.append("user_id")

    if "merchant_category" in selected:
        output["merchant_category"] = frame[selected["merchant_category"]].astype(str).str.strip()
    else:
        output["merchant_category"] = "unknown"
        inferred.append("merchant_category")

    if "device_id" in selected:
        output["device_id"] = frame[selected["device_id"]].astype(str).str.strip()
    else:
        output["device_id"] = "inferred-device-" + output["user_id"]
        inferred.append("device_id")

    if "card_id" in selected:
        output["card_id"] = frame[selected["card_id"]].astype(str).str.strip()
    else:
        output["card_id"] = (
            "inferred-card-" + output["user_id"] + "-" + output["merchant_category"]
        )
        inferred.append("card_id")

    if "billing_country" in selected:
        output["billing_country"] = frame[selected["billing_country"]].astype(str).str.strip().str.upper()
    else:
        output["billing_country"] = "UNKNOWN"
        inferred.append("billing_country")

    if "ip_country" in selected:
        output["ip_country"] = frame[selected["ip_country"]].astype(str).str.strip().str.upper()
    elif "ip_address" in selected:
        output["ip_country"] = [
            _ip_country_from_address(ip_address, billing_country)
            for ip_address, billing_country in zip(
                frame[selected["ip_address"]], output["billing_country"], strict=True
            )
        ]
        mapped[str(selected["ip_address"])] = "ip_country"
    elif "ip_billing" in selected:
        ip_billing = frame[selected["ip_billing"]].map(_normalize_ip_billing)
        output["ip_country"] = [
            billing
            if _ip_billing_mismatch(value) is not True
            else "EXTERNAL"
            for value, billing in zip(ip_billing, output["billing_country"], strict=True)
        ]
    else:
        output["ip_country"] = output["billing_country"]
        inferred.append("ip_country")

    required_order = [
        "transaction_id", "timestamp", "user_id", "device_id", "card_id", "amount",
        "billing_country", "ip_country", "merchant_category",
    ]
    output["currency"] = currency
    if "uploaded_velocity_per_hour" in selected:
        output["uploaded_velocity_per_hour"] = frame[selected["uploaded_velocity_per_hour"]].map(
            _parse_velocity_per_hour
        )
    if "ip_address" in selected:
        output["ip_address"] = frame[selected["ip_address"]].astype(str).str.strip()
    if "ip_billing" in selected:
        output["ip_billing"] = frame[selected["ip_billing"]].map(_normalize_ip_billing)
    if "device_id" in selected:
        device_values = frame[selected["device_id"]].astype(str).str.strip()
        device_statuses = device_values.map(
            lambda value: (
                "New" if value.casefold() in {"new", "new device", "first seen"}
                else "Known" if value.casefold() in {"known", "known device", "existing"}
                else None
            )
        )
        # A source column containing New/Known is a display signal, not a
        # device identifier. Preserve it separately while retaining the
        # existing device_id mapping for model compatibility.
        if device_statuses.notna().all():
            output["device"] = device_statuses
    if "amount_deviation" in selected:
        output["amount_deviation"] = frame[selected["amount_deviation"]].map(_parse_optional_number)
    if "hour" in selected:
        output["hour"] = frame[selected["hour"]].map(_parse_hour)
    if "actual" in selected:
        output["actual"] = frame[selected["actual"]].map(_normalize_actual)
    optional_order = [
        column for column in (
            "currency", "uploaded_velocity_per_hour", "ip_address", "ip_billing", "device",
            "amount_deviation", "hour", "actual",
        )
        if column in output.columns
    ]
    ignored = [str(column) for column in frame.columns if str(column) not in mapped]
    return {
        "transactions": output.loc[:, [*required_order, *optional_order]].reset_index(drop=True),
        "mapped": mapped,
        "inferred": inferred,
        "ignored": ignored,
    }


def uploaded_scores_to_dashboard_transactions(
    scored: pd.DataFrame,
    *,
    review_threshold: float = .23,
) -> pd.DataFrame:
    """Adapt a scored upload to the transaction explorer's display contract."""
    required = {"transaction_id", "timestamp", "user_id", "amount", "score", "flagged"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"scored upload is missing: {', '.join(sorted(missing))}")
    result = pd.DataFrame(index=scored.index)
    result["payment_id"] = scored["transaction_id"].astype(str)
    result["created_at"] = pd.to_datetime(scored["timestamp"], utc=True, errors="raise")
    result["amount"] = pd.to_numeric(scored["amount"], errors="raise")
    result["currency"] = scored.get("currency", pd.Series("INR", index=scored.index)).fillna("INR")
    result["status"] = scored["flagged"].map({True: "flagged", False: "scored"})
    result["method"] = scored.get(
        "merchant_category", pd.Series("unknown", index=scored.index)
    ).astype(str)
    result["order_id"] = ""
    result["email"] = scored["user_id"].astype(str)
    result["contact"] = ""
    result["international"] = result["currency"].ne("INR")

    def first_available(*columns: str) -> pd.Series:
        for column in columns:
            if column in scored.columns:
                candidate = scored[column]
                if candidate.notna().any():
                    return candidate
        return pd.Series([None] * len(scored), index=scored.index)

    result["velocity"] = first_available(
        "velocity", "uploaded_velocity_per_hour", "card_txn_count_1h"
    )
    result["ip_billing"] = first_available("ip_billing")
    result["ip_billing_mismatch"] = first_available("geo_mismatch")
    result["new_device"] = first_available("is_new_device")
    result["device"] = first_available("device")
    result.loc[
        result["ip_billing"].isna() & result["ip_billing_mismatch"].notna(),
        "ip_billing",
    ] = result.loc[
        result["ip_billing"].isna() & result["ip_billing_mismatch"].notna(),
        "ip_billing_mismatch",
    ].map({1.0: "Mismatch", 0.0: "Match", True: "Mismatch", False: "Match"})
    result.loc[
        result["device"].isna() & result["new_device"].notna(), "device"
    ] = result.loc[
        result["device"].isna() & result["new_device"].notna(), "new_device"
    ].map({1.0: "New", 0.0: "Known", True: "New", False: "Known"})
    result["amount_deviation"] = first_available(
        "amount_deviation", "user_amount_zscore"
    )
    result["hour"] = first_available("hour")
    result.loc[result["hour"].isna(), "hour"] = result.loc[
        result["hour"].isna(), "created_at"
    ].dt.hour
    result["risk_score"] = pd.to_numeric(scored["score"], errors="raise")
    flagged = scored["flagged"].astype(bool)
    result["risk_status"] = result["risk_score"].map(
        lambda score: "Review" if score >= review_threshold else "Low risk"
    )
    result.loc[flagged, "risk_status"] = "High risk"
    result["actual"] = first_available("actual")
    result["model_status"] = first_available("status")
    result["reasons"] = scored.get("reasons", pd.Series([[] for _ in range(len(scored))], index=scored.index))
    return result.sort_values("created_at", ascending=False, ignore_index=True)


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
    """Load held-out threshold sensitivity data for descriptive policy analysis."""
    curve = pd.read_csv(path)
    required = {
        "threshold", "precision", "recall", "f1", "false_positive_rate",
        "tp", "fp", "tn", "fn",
    }
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"threshold curve requires columns: {', '.join(sorted(missing))}")
    return curve


def cost_curve_for_ratio(curve: pd.DataFrame, cost_fp: float, cost_fn: float) -> pd.DataFrame:
    """Recompute descriptive held-out cost for a chosen error-cost assumption."""
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


def _merge_score_results(ordered: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    """Merge every field returned by the backend without losing upload metadata."""
    result = ordered.copy()
    response_columns = list(dict.fromkeys(
        column for row in results for column in row.keys()
    ))
    for column in response_columns:
        if column == "timestamp":
            # Keep the timezone-aware timestamp parsed from the upload. The
            # backend's ISO string is equivalent but would change the local
            # DataFrame dtype for downstream display and sorting.
            continue
        result[column] = [row.get(column) for row in results]
    return result


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
    return _merge_score_results(ordered, results)


def score_and_save_uploaded_dataset(
    transactions: pd.DataFrame,
    filename: str,
    api_url: str,
    timeout: float = 120.0,
    http_client=requests,
) -> dict:
    """Score one upload through FastAPI and persist the rows through its server-side store."""
    ordered = chronological_transactions(transactions)
    payload = ordered.copy()
    payload["timestamp"] = payload["timestamp"].map(lambda value: value.isoformat())
    try:
        response = http_client.post(
            f"{api_url.rstrip('/')}/datasets/score",
            json={"filename": filename, "transactions": payload.to_dict(orient="records")},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("Dataset scoring timed out. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                pass
        raise ScoringAPIError(detail or "The dataset could not be scored and saved.") from exc
    except ValueError as exc:
        raise ScoringAPIError("The scoring service returned an invalid dataset response.") from exc

    required_body = {
        "dataset_id", "filename", "row_count", "results",
        "signal_importance_percent", "signal_support_percent", "decision_threshold",
    }
    required_result = {"score", "flagged", "blocked", "reasons"}
    if not isinstance(body, dict) or not required_body <= body.keys():
        raise ScoringAPIError("The scoring service returned an invalid dataset response.")
    results = body["results"]
    if (
        not isinstance(results, list)
        or len(results) != len(ordered)
        or any(not isinstance(row, dict) or not required_result <= row.keys() for row in results)
    ):
        raise ScoringAPIError("The scoring service returned incomplete dataset results.")
    signal_importance = body["signal_importance_percent"]
    if not isinstance(signal_importance, dict) or not signal_importance:
        raise ScoringAPIError(
            "The scoring service returned no dataset signal importance. Restart the backend and try again."
        )
    try:
        signal_importance = {
            str(signal): float(percent) for signal, percent in signal_importance.items()
        }
    except (TypeError, ValueError) as exc:
        raise ScoringAPIError("The scoring service returned invalid dataset signal importance.") from exc
    signal_support = body["signal_support_percent"]
    if not isinstance(signal_support, dict):
        raise ScoringAPIError("The scoring service returned invalid dataset signal coverage.")
    try:
        signal_support = {
            str(signal): float(percent) for signal, percent in signal_support.items()
        }
    except (TypeError, ValueError) as exc:
        raise ScoringAPIError("The scoring service returned invalid dataset signal coverage.") from exc

    scored = _merge_score_results(ordered, results)
    dataset_id = body["dataset_id"]
    return {
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "filename": str(body["filename"]),
        "row_count": int(body["row_count"]),
        "scored": scored,
        "storage_status": str(body.get("storage_status", "saved")),
        "storage_error": body.get("storage_error"),
        "signal_importance_percent": signal_importance,
        "signal_support_percent": signal_support,
        "decision_threshold": float(body["decision_threshold"]),
    }


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


def _request_error_detail(exc: requests.RequestException) -> str:
    if exc.response is None:
        return ""
    try:
        return exc.response.json().get("detail", "")
    except ValueError:
        return ""


def list_fraud_cases(
    api_url: str,
    *,
    status: str | None = None,
    timeout: float = 15.0,
    http_client=requests,
) -> list[dict]:
    """List case-management rows, optionally filtered by status."""
    try:
        response = http_client.get(
            f"{api_url.rstrip('/')}/cases",
            params={"status": status} if status else None,
            timeout=timeout,
        )
        response.raise_for_status()
        cases = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("Listing cases timed out. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        raise ScoringAPIError(_request_error_detail(exc) or "Cases could not be loaded.") from exc
    except ValueError as exc:
        raise ScoringAPIError("The scoring service returned an invalid case list.") from exc
    if not isinstance(cases, list):
        raise ScoringAPIError("The scoring service returned an invalid case list.")
    return cases


def get_fraud_case(
    transaction_id: str,
    api_url: str,
    *,
    timeout: float = 15.0,
    http_client=requests,
) -> dict:
    """Fetch one case's current status plus its notes."""
    try:
        response = http_client.get(
            f"{api_url.rstrip('/')}/cases/{transaction_id}", timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("Loading the case timed out. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        raise ScoringAPIError(_request_error_detail(exc) or "The case could not be loaded.") from exc
    except ValueError as exc:
        raise ScoringAPIError("The scoring service returned an invalid case.") from exc
    if not isinstance(payload, dict) or "case" not in payload or "notes" not in payload:
        raise ScoringAPIError("The scoring service returned an invalid case.")
    return payload


def set_fraud_case_status(
    transaction_id: str,
    status: str,
    api_url: str,
    *,
    actor: str = "",
    risk_score: float | None = None,
    timeout: float = 15.0,
    http_client=requests,
) -> dict:
    """Set a case's investigation status, creating the case if it doesn't exist."""
    try:
        response = http_client.post(
            f"{api_url.rstrip('/')}/cases/{transaction_id}/status",
            json={"status": status, "actor": actor, "risk_score": risk_score},
            timeout=timeout,
        )
        response.raise_for_status()
        case = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("Saving the case status timed out. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        raise ScoringAPIError(_request_error_detail(exc) or "The case status could not be saved.") from exc
    except ValueError as exc:
        raise ScoringAPIError("The scoring service returned an invalid case.") from exc
    if not isinstance(case, dict):
        raise ScoringAPIError("The scoring service returned an invalid case.")
    return case


def add_fraud_case_note(
    transaction_id: str,
    note: str,
    api_url: str,
    *,
    author: str = "",
    timeout: float = 15.0,
    http_client=requests,
) -> dict:
    """Append an analyst note to a case, creating the case if it doesn't exist."""
    try:
        response = http_client.post(
            f"{api_url.rstrip('/')}/cases/{transaction_id}/notes",
            json={"note": note, "author": author},
            timeout=timeout,
        )
        response.raise_for_status()
        saved_note = response.json()
    except requests.Timeout as exc:
        raise ScoringAPIError("Saving the note timed out. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ScoringAPIError("The scoring service is offline or unreachable.") from exc
    except requests.RequestException as exc:
        raise ScoringAPIError(_request_error_detail(exc) or "The note could not be saved.") from exc
    except ValueError as exc:
        raise ScoringAPIError("The scoring service returned an invalid note.") from exc
    if not isinstance(saved_note, dict):
        raise ScoringAPIError("The scoring service returned an invalid note.")
    return saved_note
