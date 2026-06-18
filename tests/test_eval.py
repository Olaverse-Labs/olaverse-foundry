"""
Eval-harness tests. macro_f1 + exports are pure numpy (run anywhere); the
end-to-end evaluate/compare paths need torch and are guarded so they skip
without it.
"""
import numpy as np
import pytest


def test_macro_f1_perfect():
    from foundry import macro_f1
    p = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(p, p, 3) == 1.0


def test_macro_f1_ignores_absent_classes():
    from foundry import macro_f1
    # class 2 never appears in labels → excluded from the average
    preds   = np.array([0, 1, 0, 1])
    labels  = np.array([0, 1, 0, 1])
    assert macro_f1(preds, labels, num_labels=5) == 1.0


def test_macro_f1_partial():
    from foundry import macro_f1
    preds  = np.array([0, 1, 2, 1, 0])
    labels = np.array([0, 1, 2, 2, 0])
    f1 = macro_f1(preds, labels, 3)
    assert 0.0 < f1 < 1.0


def test_exports_present():
    import foundry
    for n in ("evaluate_encoder", "compare_encoders", "print_comparison", "macro_f1"):
        assert n in foundry.__all__


def test_print_comparison_runs(capsys):
    from foundry import print_comparison
    print_comparison({
        "Purple Mist Base": {"accuracy": 0.82, "macro_f1": 0.79, "params_m": 30.0},
        "mBERT":            {"accuracy": 0.80, "macro_f1": 0.77, "params_m": 178.0},
    })
    out = capsys.readouterr().out
    assert "Purple Mist Base" in out and "macro_f1" in out


def test_compare_encoders_handles_failure_gracefully():
    # A bogus model id should be caught and recorded as NaN, not raise.
    from foundry import compare_encoders
    res = compare_encoders({"bogus": "definitely/not-a-real-model-xyz"},
                           [{"text": "hi", "label": 0}],
                           [{"text": "hi", "label": 0}],
                           num_labels=2, task="sequence")
    assert "bogus" in res
    assert res["bogus"]["macro_f1"] != res["bogus"]["macro_f1"]  # NaN


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
