from data.generate_synthetic import generate_transactions


def test_generator_is_reproducible_imbalanced_and_well_formed():
    first = generate_transactions(2_000, seed=17)
    second = generate_transactions(2_000, seed=17)
    assert first.equals(second)
    assert 0.01 <= first["is_fraud"].mean() <= 0.02
    assert first["transaction_id"].is_unique
    assert first["timestamp"].is_monotonic_increasing
    assert (first["amount"] >= 0).all()
