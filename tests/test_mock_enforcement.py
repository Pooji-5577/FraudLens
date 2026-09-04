from pathlib import Path


def test_mock_enforcement_module_has_no_real_gateway_or_http_dependency():
    source = Path("frontend/mock_enforcement.py").read_text()

    assert "razorpay_enforcement" not in source
    assert "requests" not in source
