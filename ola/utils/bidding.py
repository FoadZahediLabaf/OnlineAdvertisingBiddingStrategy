"""
Bidding utility functions for the OLA project.

Owner:
    Shared utility file

Purpose:
    Store general helper functions for auction outcomes, rewards, costs, and bid checks.

Do not:
    - Add algorithm-specific logic here.
    - Add experiment-running code here.
    - Hard-code parameters for a specific requirement.
    - Change utility behavior without telling the group.
"""


def wins_auction(bid: float, competing_bid: float) -> bool:
    """Return True if the bid wins a first-price auction."""
    return bid >= competing_bid


def first_price_reward(value: float, bid: float, competing_bid: float) -> float:
    """Return reward for a first-price auction with one campaign."""
    if wins_auction(bid, competing_bid):
        return value - bid
    return 0.0


def first_price_cost(bid: float, competing_bid: float) -> float:
    """Return cost for a first-price auction with one campaign."""
    if wins_auction(bid, competing_bid):
        return bid
    return 0.0


def is_affordable(bid: float, budget_left: float) -> bool:
    """Return True if the bid can be paid with the remaining budget."""
    return bid <= budget_left
