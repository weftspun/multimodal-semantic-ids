"""Stdlib-only tests for eval metrics. Run: ``python -m tests.test_metrics``."""

from vsk_recsys.eval.metrics import mrr_at_k, recall_at_k


def main() -> None:
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["p", "q", "r"]]
    truth = ["b", "z", "nope"]

    # Recall@K: b hits (rank2), z hits (rank3), nope misses -> 2/3 at k=3; none at k=1.
    assert abs(recall_at_k(ranked, truth, 3) - 2 / 3) < 1e-9
    assert abs(recall_at_k(ranked, truth, 1) - 0.0) < 1e-9

    # MRR@K: (1/2 + 1/3 + 0) / 3 at k=3; 0 at k=1.
    assert abs(mrr_at_k(ranked, truth, 3) - (0.5 + 1 / 3) / 3) < 1e-9
    assert abs(mrr_at_k(ranked, truth, 1) - 0.0) < 1e-9

    # Set-valued ground truth (any acceptable next item).
    assert abs(recall_at_k([["a", "b"]], [{"b", "z"}], 2) - 1.0) < 1e-9

    # Edge cases.
    assert recall_at_k([], [], 5) == 0.0
    try:
        recall_at_k([["a"]], ["a", "b"], 3)
        raise AssertionError("expected length-mismatch ValueError")
    except ValueError:
        pass

    print("test_metrics: OK")


if __name__ == "__main__":
    main()
