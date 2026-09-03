from urllib.parse import parse_qs, urlparse

from dashboard.razorpay_oauth import (
    PAYMENTS_URL,
    REVOKE_URL,
    TOKEN_URL,
    amount_from_subunits,
    build_authorization_url,
    consume_oauth_state,
    exchange_authorization_code,
    fetch_payments,
    issue_oauth_state,
    revoke_access_token,
    state_matches,
)


def test_authorization_url_requests_read_only_access_and_state():
    url = build_authorization_url("client_123", "https://app.test/callback", "csrf-state")
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.netloc == "auth.razorpay.com"
    assert query == {
        "client_id": ["client_123"],
        "response_type": ["code"],
        "redirect_uri": ["https://app.test/callback"],
        "scope": ["read_only"],
        "state": ["csrf-state"],
    }
    assert state_matches("csrf-state", "csrf-state")
    assert not state_matches("csrf-state", "different-state")
    assert not state_matches(None, "csrf-state")

    issued = issue_oauth_state()
    assert consume_oauth_state(issued)
    assert not consume_oauth_state(issued)


def test_authorization_code_is_exchanged_server_side():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "access", "razorpay_account_id": "acc_123"}

    class Client:
        def post(self, url, json, timeout):
            self.url = url
            self.payload = json
            self.timeout = timeout
            return Response()

    client = Client()
    token = exchange_authorization_code(
        "auth-code",
        client_id="client_123",
        client_secret="server-secret",
        redirect_uri="https://app.test/callback",
        mode="test",
        http_client=client,
    )

    assert client.url == TOKEN_URL
    assert client.payload == {
        "client_id": "client_123",
        "client_secret": "server-secret",
        "grant_type": "authorization_code",
        "redirect_uri": "https://app.test/callback",
        "code": "auth-code",
        "mode": "test",
    }
    assert token["razorpay_account_id"] == "acc_123"


def test_payments_are_fetched_with_bearer_auth_and_date_filters():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"items": [{"id": "pay_123"}]}

    class Client:
        def get(self, url, headers, params, timeout):
            self.request = (url, headers, params, timeout)
            return Response()

    client = Client()
    payments = fetch_payments(
        "secret-access-token",
        from_timestamp=1_700_000_000,
        to_timestamp=1_700_086_399,
        http_client=client,
    )

    assert payments == [{"id": "pay_123"}]
    assert client.request == (
        PAYMENTS_URL,
        {"Authorization": "Bearer secret-access-token"},
        {"from": 1_700_000_000, "to": 1_700_086_399, "count": 100, "skip": 0},
        15.0,
    )


def test_amount_conversion_uses_iso_4217_minor_units():
    assert amount_from_subunits(12_500, "INR") == 125.0
    assert amount_from_subunits(12_500, "USD") == 125.0
    assert amount_from_subunits(12_500, "JPY") == 12_500.0


def test_access_token_is_revoked_on_razorpay():
    class Response:
        def raise_for_status(self):
            return None

    class Client:
        def post(self, url, json, timeout):
            self.request = (url, json, timeout)
            return Response()

    client = Client()
    revoke_access_token(
        "access-token",
        client_id="client_123",
        client_secret="server-secret",
        http_client=client,
    )

    assert client.request == (
        REVOKE_URL,
        {
            "client_id": "client_123",
            "client_secret": "server-secret",
            "token_type_hint": "access_token",
            "token": "access-token",
        },
        15.0,
    )
