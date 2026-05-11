from collections import deque

from etic.core.data_structure.pauli import PauliChar
from etic.core.data_structure.tanner_graph import TannerGraph
from etic.core.data_structure.tanner_node import CheckNode, TannerNode


class GraphAlgorithm:
    """Namespace for graph algorithms operating on Tanner graphs."""

    @staticmethod
    def shortest_path(
        graph: TannerGraph,
        start: TannerNode,
        end: TannerNode,
        check_type: PauliChar | None = None,
    ) -> list[TannerNode]:
        """
        Returns the shortest path between two nodes in the Tanner graph using breadth-first search (BFS).
        The path is returned as a list of nodes, from `start` node to `end` node.
        If no path exists, an empty list is returned.

        Parameters
        ----------
        check_type : PauliChar | None
            When set, only check nodes whose ``check_type`` matches this value
            may appear in the path.  Variable nodes are always allowed.
        """
        if start not in graph.variable_nodes and start not in graph.check_nodes:
            raise ValueError("Start node is not in the Tanner graph.")
        if end not in graph.variable_nodes and end not in graph.check_nodes:
            raise ValueError("End node is not in the Tanner graph.")

        queue = deque([start])
        visited = {start}  # type: ignore
        parent = {start: None}  # type: ignore

        while queue:
            current = queue.popleft()

            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]  # type: ignore

            for neighbor in graph.get_neighbourhood(current):  # type: ignore
                if neighbor not in visited:
                    if (
                        check_type is not None
                        and isinstance(neighbor, CheckNode)
                        and neighbor.check_type != check_type
                    ):
                        continue
                    visited.add(neighbor)
                    parent[neighbor] = current  # type: ignore
                    queue.append(neighbor)

        return []  # type: ignore
