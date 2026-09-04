"""Server-side helpers for the Razorpay OAuth authorization-code flow."""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from typing import Any
from urllib.parse import urlencode

import requests


AUTHORIZE_URL = "https://auth.razorpay.com/authorize"
TOKEN_URL = "https://auth.razorpay.com/token"
REVOKE_URL = "https://auth.razorpay.com/revoke"
PAYMENTS_URL = "https://api.razorpay.com/v1/payments"
_PENDING_STATES: dict[str, float] = {}
_STATE_LOCK = threading.Lock()


class RazorpayOAuthError(RuntimeError):
    """A safe, user-displayable Razorpay connection failure."""


# ISO 4217 currencies that differ from the common two-minor-unit convention.
# All other codes use two minor units. This covers Razorpay's documented zero-
# and three-decimal payment currencies without treating every amount as paise.
_ZERO_MINOR_UNIT_CURRENCIES = {
    "BIF", "CLP", "DJF", "GNF", "ISK", "JPY", "KMF", "KRW", "PYG",
    "RWF", "UGX", "UYI", "VND", "VUV", "XAF", "XOF", "XPF",
}
_THREE_MINOR_UNIT_CURRENCIES = {"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"}
_FOUR_MINOR_UNIT_CURRENCIES = {"CLF", "UYW"}


def amount_from_subunits(amount: int | float, currency: str) -> float:
    """Convert a Razorpay integer amount using ISO 4217 minor units."""
    code = str(currency).upper()
    minor_units = (
        0 if code in _ZERO_MINOR_UNIT_CURRENCIES
        else 3 if code in _THREE_MINOR_UNIT_CURRENCIES
        else 4 if code in _FOUR_MINOR_UNIT_CURRENCIES
        else 2
    )
    return float(amount) / (10 ** minor_units)


def build_authorization_url(client_id: str, redirect_uri: str, state: str) -> str:
    """Build the Razorpay OAuth URL needed for human-approved enforcement."""
    parameters = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": "read_write",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(parameters)}"


def state_matches(expected: str | None, received: str | None) -> bool:
    """Compare OAuth state values without leaking timing information."""
    if not expected or not received:
        return False
    return hmac.compare_digest(expected, received)


def issue_oauth_state(ttl_seconds: int = 600) -> str:
    """Issue a one-use state that survives the browser's trip through Razorpay."""
    state = secrets.token_urlsafe(32)
    now = time.monotonic()
    with _STATE_LOCK:
        expired = [key for key, deadline in _PENDING_STATES.items() if deadline <= now]
        for key in expired:
            _PENDING_STATES.pop(key, None)
        _PENDING_STATES[state] = now + ttl_seconds
    return state


def consume_oauth_state(received: str | None) -> bool:
    """Validate and invalidate a pending OAuth state value."""
    if not received:
        return False
    now = time.monotonic()
    with _STATE_LOCK:
        deadline = _PENDING_STATES.pop(received, None)
    return deadline is not None and deadline > now


def exchange_authorization_code(
    code: str,
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    mode: str = "test",
    timeout: float = 15.0,
    http_client=requests,
) -> dict[str, Any]:
    """Exchange an authorization code from Razorpay on the server."""
    if mode != "test":
        raise RazorpayOAuthError("FraudLens Razorpay access is restricted to Test Mode.")
    try:
        response = http_client.post(
            TOKEN_URL,
            json={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
                "code": code,
                "mode": mode,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        token = response.json()
    except requests.Timeout as exc:
        raise RazorpayOAuthError("Razorpay connection timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise RazorpayOAuthError("Razorpay rejected the account connection. Please try again.") from exc
    except ValueError as exc:
        raise RazorpayOAuthError("Razorpay returned an invalid connection response.") from exc

    if not isinstance(token, dict) or not token.get("access_token"):
        raise RazorpayOAuthError("Razorpay did not return an access token.")
    token["mode"] = "test"
    return token


def revoke_access_token(
    access_token: str,
    *,
    client_id: str,
    client_secret: str,
    timeout: float = 15.0,
    http_client=requests,
) -> None:
    """Revoke an OAuth access token at Razorpay before disconnecting locally."""
    try:
        response = http_client.post(
            REVOKE_URL,
            json={
                "client_id": client_id,
                "client_secret": client_secret,
                "token_type_hint": "access_token",
                "token": access_token,
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RazorpayOAuthError("Razorpay disconnection timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise RazorpayOAuthError("Razorpay access could not be revoked. Please try again.") from exc


def fetch_payments(
    access_token: str,
    *,
    from_timestamp: int,
    to_timestamp: int,
    limit: int | None = None,
    timeout: float = 15.0,
    http_client=requests,
) -> list[dict[str, Any]]:
    """Fetch every payment in a date range, paging in batches of 100."""
    items: list[dict[str, Any]] = []
    skip = 0
    while limit is None or len(items) < limit:
        count = 100 if limit is None else min(100, limit - len(items))
        try:
            response = http_client.get(
                PAYMENTS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "from": int(from_timestamp),
                    "to": int(to_timestamp),
                    "count": count,
                    "skip": skip,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise RazorpayOAuthError("Razorpay payments timed out. Please try again.") from exc
        except requests.RequestException as exc:
            raise RazorpayOAuthError("Razorpay payments could not be loaded.") from exc
        except ValueError as exc:
            raise RazorpayOAuthError("Razorpay returned an invalid payments response.") from exc

        page = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(page, list):
            raise RazorpayOAuthError("Razorpay returned an invalid payments response.")
        items.extend(item for item in page if isinstance(item, dict))
        if len(page) < count:
            break
        skip += count
    return items
