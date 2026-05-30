from hybrid_american_pricer.experiments.run_benchmarks import run_stage1_benchmarks


def test_stage1_benchmarks_include_required_comparisons() -> None:
    results = run_stage1_benchmarks()
    assert {
        "case_id",
        "method",
        "price",
        "absolute_error",
        "relative_error",
        "runtime_seconds",
    }.issubset(results.columns)
    assert "Black-Scholes vs Binomial Tree" in set(results["comparison"])
    assert "Monte Carlo vs Black-Scholes" in set(results["comparison"])
    assert "American Put vs European Put" in set(results["comparison"])
    assert (results["price"] > 0).all()

