"""
Ensemble models module - Combine multiple models for improved predictions.

This module provides:
- StackingEnsemble: Combines base models with meta-learner
- VotingEnsemble: Simple averaging of predictions
- BlendingEnsemble: Holdout-based meta-learning
"""

from src.models.ensemble.stacking import (
    StackingEnsemble,
    VotingEnsemble,
    BlendingEnsemble,
    create_ensemble,
)

__all__ = [
    "StackingEnsemble",
    "VotingEnsemble",
    "BlendingEnsemble",
    "create_ensemble",
]
