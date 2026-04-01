from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import networkx as nx
import numpy as np


def analyze_grid_topology(
    obs,
    line_or_to_subid: Sequence[int],
    line_ex_to_subid: Sequence[int],
    n_sub: int,
) -> dict[str, Any]:
    """Build topology intelligence from a raw Grid2Op observation."""

    energy_graph = obs.get_energy_graph()
    line_status = [bool(x) for x in obs.line_status.tolist()]
    rho = [float(x) for x in obs.rho.tolist()]
    overflow = [int(x) for x in obs.timestep_overflow.tolist()]
    connected_line_ids = [line_id for line_id, status in enumerate(line_status) if status]

    bus_graph = nx.MultiGraph()
    for sub_id in range(n_sub):
        bus_graph.add_node(sub_id)
    for line_id in connected_line_ids:
        u = int(line_or_to_subid[line_id])
        v = int(line_ex_to_subid[line_id])
        bus_graph.add_edge(
            u,
            v,
            key=line_id,
            line_id=line_id,
            rho=rho[line_id],
            timestep_overflow=overflow[line_id],
        )

    active_bus_graph = nx.Graph()
    active_bus_graph.add_nodes_from(
        node for node in bus_graph.nodes if bus_graph.degree(node) > 0
    )
    for u, v, data in bus_graph.edges(data=True):
        if active_bus_graph.has_edge(u, v):
            continue
        active_bus_graph.add_edge(u, v, rho=float(data["rho"]))

    pair_to_lines: dict[tuple[int, int], list[int]] = defaultdict(list)
    for line_id in connected_line_ids:
        u = int(line_or_to_subid[line_id])
        v = int(line_ex_to_subid[line_id])
        key = tuple(sorted((int(u), int(v))))
        pair_to_lines[key].append(line_id)

    parallel_groups = {
        str(line_id): sorted(other for other in line_ids if other != line_id)
        for line_ids in pair_to_lines.values()
        if len(line_ids) > 1
        for line_id in line_ids
    }

    bridge_lines: list[int] = []
    safe_to_disconnect: list[int] = []
    for line_id in connected_line_ids:
        trial_graph = bus_graph.copy()
        u = int(line_or_to_subid[line_id])
        v = int(line_ex_to_subid[line_id])
        if not trial_graph.has_edge(u, v, key=line_id):
            continue
        trial_graph.remove_edge(u, v, key=line_id)
        active_nodes = [node for node in trial_graph.nodes if trial_graph.degree(node) > 0]
        if not active_nodes:
            bridge_lines.append(line_id)
            continue
        reduced_graph = nx.Graph(trial_graph.subgraph(active_nodes))
        if nx.number_connected_components(reduced_graph) > 1:
            bridge_lines.append(line_id)
        else:
            safe_to_disconnect.append(line_id)

    components = [
        sorted(component)
        for component in nx.connected_components(active_bus_graph)
    ] if active_bus_graph.number_of_nodes() else []
    islanded_clusters = components[1:] if len(components) > 1 else []

    centrality_graph = active_bus_graph.copy()
    centrality_scores = (
        nx.betweenness_centrality(centrality_graph)
        if centrality_graph.number_of_nodes() > 0
        else {}
    )
    high_centrality_buses = [
        int(node)
        for node, score in sorted(
            centrality_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if score > 0.0
    ][:3]

    flow_matrix = _extract_flow_matrix(obs.flow_bus_matrix(active_flow=True))
    exporter_buses, importer_buses = _rank_flow_buses(flow_matrix)
    stressed_lines = [
        {
            "line_id": line_id,
            "rho": round(rho[line_id], 4),
            "overflow": overflow[line_id],
            "from_sub": int(line_or_to_subid[line_id]),
            "to_sub": int(line_ex_to_subid[line_id]),
        }
        for line_id in sorted(connected_line_ids, key=lambda idx: rho[idx], reverse=True)[:5]
    ]

    congestion_corridor = "none"
    if stressed_lines:
        corridor_lines = [entry["line_id"] for entry in stressed_lines[:3]]
        congestion_corridor = (
            f"export buses {exporter_buses or ['unknown']} -> "
            f"import buses {importer_buses or ['unknown']} via lines {corridor_lines}"
        )

    return {
        "num_buses": int(active_bus_graph.number_of_nodes()),
        "num_connected_lines": len(connected_line_ids),
        "bridge_line_count": len(bridge_lines),
        "n1_security_score": round(
            max(0.0, 1.0 - (len(bridge_lines) / max(1, len(connected_line_ids)))),
            6,
        ),
        "bridge_lines": sorted(bridge_lines),
        "safe_to_disconnect": sorted(safe_to_disconnect),
        "n_minus_1_critical_lines": sorted(bridge_lines),
        "parallel_groups": parallel_groups,
        "high_centrality_buses": high_centrality_buses,
        "islanded_clusters": islanded_clusters,
        "congestion_corridor": congestion_corridor,
        "flow_clusters": {
            "export_buses": exporter_buses,
            "import_buses": importer_buses,
        },
        "stressed_lines": stressed_lines,
        "graph_density": round(nx.density(active_bus_graph), 6)
        if active_bus_graph.number_of_nodes() > 1
        else 0.0,
        "energy_graph_summary": {
            "nodes": energy_graph.number_of_nodes(),
            "edges": energy_graph.number_of_edges(),
        },
    }


def _extract_flow_matrix(raw_flow_output) -> np.ndarray:
    if isinstance(raw_flow_output, np.ndarray):
        return raw_flow_output
    if isinstance(raw_flow_output, tuple) and raw_flow_output:
        first = raw_flow_output[0]
        if isinstance(first, np.ndarray):
            return first
    raise TypeError(f"Unsupported flow_bus_matrix output type: {type(raw_flow_output)!r}")


def _rank_flow_buses(flow_matrix: np.ndarray) -> tuple[list[int], list[int]]:
    matrix = np.asarray(flow_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square flow matrix, got shape {matrix.shape}")

    # In Grid2Op's flow_bus_matrix, the diagonal carries the nodal net active injection
    # while off-diagonal entries represent inter-bus transfers. Summing rows therefore
    # tends to zero by construction and hides the exporter/importer ranking.
    net_exports = np.diag(matrix)
    exporters = [
        int(idx)
        for idx in np.argsort(-net_exports)
        if math.isfinite(net_exports[idx]) and net_exports[idx] > 1.0
    ][:3]
    importers = [
        int(idx)
        for idx in np.argsort(net_exports)
        if math.isfinite(net_exports[idx]) and net_exports[idx] < -1.0
    ][:3]
    return exporters, importers
