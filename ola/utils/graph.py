"""
Graph utilities for campaign compatibility constraints.

Owner:
    Shared utility file

Shared with:
    Requirement 2, Requirement 3, and Requirement 4

Purpose:
    Store helper functions for conflict graphs between campaigns.

Do not:
    - Add experiment-specific logic here.
    - Hard-code campaign names.
    - Change existing function behavior without telling the group.
"""

from typing import Iterable, Set, Tuple

import networkx as nx


ConflictEdge = Tuple[int, int]


def build_conflict_graph(number_of_campaigns: int, edges: Iterable[ConflictEdge]) -> nx.Graph:
    """Build a conflict graph where nodes are campaigns and edges mean incompatibility."""
    if number_of_campaigns <= 0:
        raise ValueError("number_of_campaigns must be positive.")

    validated_edges = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Each conflict edge must contain exactly two campaign indices.")

        campaign_a, campaign_b = edge

        if campaign_a == campaign_b:
            raise ValueError("Conflict edge cannot connect a campaign to itself.")

        if not (0 <= campaign_a < number_of_campaigns):
            raise ValueError("Conflict edge contains campaign index out of range.")

        if not (0 <= campaign_b < number_of_campaigns):
            raise ValueError("Conflict edge contains campaign index out of range.")

        validated_edges.append((campaign_a, campaign_b))

    graph = nx.Graph()
    graph.add_nodes_from(range(number_of_campaigns))
    graph.add_edges_from(validated_edges)
    return graph

def is_feasible_campaign_set(campaigns: Iterable[int], conflict_graph: nx.Graph) -> bool:
    """Return True if no two selected campaigns conflict with each other."""
    selected_campaigns: Set[int] = set(campaigns)
    for campaign_a, campaign_b in conflict_graph.edges:
        if campaign_a in selected_campaigns and campaign_b in selected_campaigns:
            return False
    return True
