"""Phase 1 — MovieLens parity: FSQ semantic IDs + Transformer next-item generation.

Proves the FOSS stack green against a PinSage Recall@K / MRR@K baseline before touching Godot data.
Item features = title text (ModernBERT) + genres; uses RQ-VAE-Recommender's MovieLens auto-download.

Implemented in Phase 1 (planner actions a_p1_data / a_p1_fsq / a_p1_decoder / a_p1_train / a_p1_eval).
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Phase 1 MovieLens training is not implemented yet "
        "(taskweft actions a_p1_data..a_p1_parity)."
    )


if __name__ == "__main__":
    main()
