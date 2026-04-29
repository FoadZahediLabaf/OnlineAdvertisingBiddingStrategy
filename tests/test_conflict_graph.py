"""Tests for shared conflict-graph utilities."""

from ola.utils.graph import build_conflict_graph, is_feasible_campaign_set


def test_build_conflict_graph_adds_all_campaigns_and_edges() -> None:
    graph = build_conflict_graph(number_of_campaigns=4, edges=[(0, 1), (2, 3)])

    assert set(graph.nodes) == {0, 1, 2, 3}
    assert graph.has_edge(0, 1)
    assert graph.has_edge(2, 3)
    assert not graph.has_edge(0, 2)


def test_is_feasible_campaign_set_detects_conflicts() -> None:
    graph = build_conflict_graph(number_of_campaigns=3, edges=[(0, 1)])

    assert is_feasible_campaign_set([0, 2], graph) is True
    assert is_feasible_campaign_set([0, 1], graph) is False
    assert is_feasible_campaign_set([], graph) is True