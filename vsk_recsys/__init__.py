"""FOSS semantic-ID generative-retrieval recommender for session-based Godot asset recommendation.

Migrated off LibRecommender/PinSage. Architecture and rationale: decisions/20260712-*.md.
Kept import-light on purpose so the stdlib-only modules (data.etnf, eval.metrics) import
without torch/numpy present.
"""

__version__ = "0.1.0"
