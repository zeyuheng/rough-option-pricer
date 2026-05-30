from hybrid_american_pricer.experiments.lsmc_study import (
    run_basis_comparison,
    run_lsmc_convergence,
    summarize_stage2,
)


def test_lsmc_study_outputs_required_columns() -> None:
    convergence = run_lsmc_convergence(path_counts=(200, 400), time_steps=(10,), seed=7)
    basis = run_basis_comparison(bases=("polynomial", "laguerre"), n_paths=400, n_steps=10, seed=7)
    assert {
        "n_paths",
        "n_steps",
        "lsmc_price",
        "absolute_error",
        "relative_error",
        "runtime_seconds",
        "mean_regression_condition",
    }.issubset(convergence.columns)
    assert {"polynomial", "laguerre"} == set(basis["basis"])
    summary = summarize_stage2(convergence, basis)
    assert "best_basis" in summary
    assert "most_stable_basis" in summary

