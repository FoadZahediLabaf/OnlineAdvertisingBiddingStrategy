"""
Base environment interface for the OLA project.

Owner:
    Foad

Purpose:
    Define the common interface that all environments should follow.

Do not:
    - Add requirement-specific logic here.
    - Hard-code parameters for a specific experiment.
    - Change method names without group approval.
    - Change the step output format without group approval.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


STEP_OUTPUT_KEYS: Tuple[str, ...] = (
    "round",
    "action",
    "bids",
    "wins",
    "reward",
    "cost",
    "budget_left",
    "done",
    "info",
)


class BaseEnvironment(ABC):
    """Abstract base class for all project environments."""

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the environment before a new experiment run."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Dict[str, Any]:
        """
        Execute one interaction round.

        Expected output keys when available are defined in STEP_OUTPUT_KEYS:
            - round
            - action
            - bids
            - wins
            - reward
            - cost
            - budget_left
            - done
            - info
        """
        raise NotImplementedError
