from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydrake.geometry import Rgba
from pydrake.perception import PointCloud


@dataclass(frozen=True, eq=False)
class SearchGraph:
    nodes_q: NDArray[np.float64]
    group: NDArray[np.int8]
    parent: NDArray[np.int32]
    edges: NDArray[np.int32] | None = None


def planner_graph(planner, si, state_to_q) -> SearchGraph:
    from ompl import base as ob, geometric as og
    data = ob.PlannerData(si)
    planner.getPlannerData(data)
    count = data.numVertices()
    bidirectional = isinstance(planner, og.RRTConnect)
    nodes = np.empty((count, si.getStateSpace().getDimension()), dtype=float)
    group = np.zeros(count, dtype=np.int8)
    for i in range(count):
        vertex = data.getVertex(i)
        nodes[i] = state_to_q(vertex.getState())
        if bidirectional and vertex.getTag() == 2:
            group[i] = 1
    parent = np.full(count, -1, dtype=np.int32)
    for i in range(count):
        edges = data.getEdges(i) if group[i] else data.getIncomingEdges(i)
        candidates = [j for j in edges if group[j] == group[i]]
        if len(candidates) > 1:
            raise ValueError("OMPL search is not a parent forest")
        if candidates:
            parent[i] = candidates[0]
    directed_edges = np.asarray([(i, j) for i in range(count) for j in data.getEdges(i)],
                                dtype=np.int32).reshape(-1, 2)
    return SearchGraph(nodes, group, parent, edges=directed_edges)


def project_search_graph(graph, project_nodes, project_path):
    if graph is None:
        return None
    edges = (graph.edges if graph.edges is not None else
             [(parent, i) for i, parent in enumerate(graph.parent) if parent >= 0])
    pairs = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    pairs = {pair for pair in pairs if pair[0] != pair[1]}
    edges = [{"points": np.asarray(project_path(graph.nodes_q[[a, b]])).tolist(),
              "group": int(graph.group[b])} for a, b in sorted(pairs)]
    return {"nodes": np.asarray(project_nodes(graph.nodes_q)).tolist(),
            "groups": graph.group.tolist(), "edges": edges,
            "node_count": len(graph.nodes_q), "edge_count": len(edges)}


def draw_search_tree(meshcat, root, graph, colors=((.157, .490, .424), (.47, .52, .50)),
                     *, z=0.):
    def xyz(points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.vstack((points.T, np.full(len(points), z)))

    for group, color in enumerate(colors):
        path = f"{root}/group{group}"
        edges = [xyz(edge["points"]) for edge in graph.get("edges", [])
                 if edge.get("group", 0) == group and len(edge["points"]) > 1]
        if edges:
            meshcat.SetLineSegments(path, np.hstack([edge[:, :-1] for edge in edges]),
                                    np.hstack([edge[:, 1:] for edge in edges]),
                                    1.5, Rgba(*color, .55))
        else:
            meshcat.Delete(path)
        nodes = [point for point, tree in zip(graph.get("nodes", []),
                                              graph.get("groups", [])) if tree == group]
        if nodes:
            cloud = PointCloud(len(nodes))
            cloud.mutable_xyzs()[:] = xyz(nodes)
            meshcat.SetObject(f"{path}/nodes", cloud, 6., Rgba(*color))
        else:
            meshcat.Delete(f"{path}/nodes")
