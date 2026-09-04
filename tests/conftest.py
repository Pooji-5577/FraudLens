"""Keep tests isolated from developer and deployment credentials in .env."""

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDLENS_STORAGE", "sqlite")
    monkeypatch.setenv(
        "RAZORPAY_ENFORCEMENT_DB", str(tmp_path / "razorpay_enforcement.sqlite3")
    )
    monkeypatch.setenv("RAZORPAY_MODE", "test")
