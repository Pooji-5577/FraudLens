"""Keep tests isolated from developer and deployment credentials in .env."""

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDLENS_STORAGE", "sqlite")
    monkeypatch.setenv(
        "RAZORPAY_ENFORCEMENT_DB", str(tmp_path / "razorpay_enforcement.sqlite3")
    )
    monkeypatch.setenv("RAZORPAY_MODE", "test")
    # Case management has no sqlite fallback, so blank real Supabase credentials
    # by default: a test that needs CaseStore must inject its own http_client or
    # explicitly set fake ones, never touch the developer's real project.
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "")
