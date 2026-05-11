from collections.abc import Callable, Mapping
from itertools import cycle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Set, cast
from uuid import UUID

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from pypdf import PdfReader, PdfWriter

from epic.core.data_structure.pauli import PauliChar
from epic.core.data_structure.tanner_graph import TannerGraph
from epic.core.data_structure.tanner_node import TannerNode

HighlightNodeRef = TannerNode | UUID | str
HighlightNodeSet = Set[HighlightNodeRef]
HighlightGroup = tuple[HighlightNodeSet, str] | tuple[HighlightNodeSet, str, str]


class TannerGraphVisualizer:
    @staticmethod
    def visualize(
        graph: TannerGraph,
        highlight_nodes: HighlightNodeSet | Iterable[HighlightGroup] | None = None,
        system_labels: (
            Mapping[tuple[int, int], str]
            | Callable[[tuple[int, int]], str | None]
            | None
        ) = None,
        output_path: str | Path | None = None,
        periodic: bool = True,
        invert_y_rows: Set[int] | None = None,
        title: str | None = None,
    ):
        """Render a Tanner graph as a Matplotlib figure or append it to a PDF.

        The layout is inferred from the node coordinate dimension shared by all
        nodes in ``graph``:

        - ``0D``: draw a bipartite layout with variables and checks separated.
        - ``2D``: draw a single planar layout using ``(x, y)`` coordinates.
        - ``3D``: draw one subplot per plane using the third coordinate as the
          plane index.
        - ``4D``: draw one subplot per system using ``(x, y, system_x,
          system_y)``, where ``system_x`` is horizontal and ``system_y`` is
          vertical in the subplot grid.

        Highlight groups can be specified with actual ``TannerNode`` instances,
        UUID objects, or UUID strings. Unmatched highlight references are listed
        in a separate legend block instead of being dropped silently.

        Parameters
        ----------
        graph : TannerGraph
            Tanner graph to render.
        highlight_nodes : HighlightNodeSet | Iterable[HighlightGroup] | None, optional
            Nodes to emphasize. Pass a single set to highlight all entries in
            gold, or an iterable of ``(node_set, color)`` or
            ``(node_set, color, label)`` tuples for multiple highlight groups.
            Use ``"auto"`` to assign colors from Matplotlib's default color
            cycle.
        system_labels : Mapping[tuple[int, int], str] | Callable[[tuple[int, int]], str | None] | None, optional
            Custom titles for 4D system subplots. When omitted, subplots are
            labeled as ``System (x, y)``.
        output_path : str | Path | None, optional
            If provided, append the rendered figure to the target PDF file and
            suppress inline display by closing the figure before returning.
        periodic : bool, optional
            When rendering 2D layouts, draw toroidal wrap-around edges
            explicitly if ``True``.
        invert_y_rows : Set[int] | None, optional
            Rows whose subplot Y axes should be inverted in 3D and 4D layouts.
        title : str | None, optional
            Override the default figure title or suptitle.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | list] | None
            The created Matplotlib figure and axes when ``output_path`` is not
            provided. Returns ``None`` after saving when ``output_path`` is set.

        Raises
        ------
        ValueError
            If node coordinates are missing inconsistently, have mixed
            dimensions, use unsupported dimensions, or contain negative 4D
            system coordinates.
        """

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        color_cycle = cycle(colors)

        pdf = None
        path = None
        temp_pdf_path = None
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                suffix=".pdf", dir=path.parent, delete=False
            ) as temp_pdf:
                temp_pdf_path = Path(temp_pdf.name)
            pdf = PdfPages(temp_pdf_path)

        def _finalize_figure(fig, axes):
            if pdf is not None:
                pdf.savefig(fig)
                pdf.close()
                assert path is not None
                assert temp_pdf_path is not None
                if path.exists():
                    writer = PdfWriter()
                    for source in (path, temp_pdf_path):
                        reader = PdfReader(str(source))
                        for page in reader.pages:
                            writer.add_page(page)
                    with path.open("wb") as merged_pdf:
                        writer.write(merged_pdf)
                    temp_pdf_path.unlink()
                else:
                    temp_pdf_path.replace(path)
                plt.close(fig)
                return None
            return fig, axes

        edge_color = {
            PauliChar.X: "tab:red",
            PauliChar.Z: "tab:blue",
            PauliChar.Y: "tab:purple",
        }

        def _plot_title(default_title: str) -> str:
            return title if title is not None else default_title

        def _resolve_highlight_color(color: str) -> str:
            return next(color_cycle) if color == "auto" else color

        def _format_unmatched_highlight(ref: HighlightNodeRef) -> str:
            if isinstance(ref, TannerNode):
                if ref.tag:
                    return f"{ref.tag} [{ref.id}]"
                return str(ref.id)
            return str(ref)

        all_nodes = list(graph.variable_nodes) + list(graph.check_nodes)
        if not all_nodes:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_title(_plot_title("Empty Tanner Graph"))
            ax.set_axis_off()
            return _finalize_figure(fig, ax)

        all_node_set = set(all_nodes)
        nodes_by_id = {node.id: node for node in all_nodes}

        def _resolve_highlight_nodes(
            refs: HighlightNodeSet,
        ) -> tuple[set[TannerNode], list[str]]:
            matched_nodes: set[TannerNode] = set()
            unmatched_refs: list[str] = []
            for ref in refs:
                matched_node = None
                if isinstance(ref, TannerNode):
                    matched_node = nodes_by_id.get(ref.id)
                elif isinstance(ref, UUID):
                    matched_node = nodes_by_id.get(ref)
                elif isinstance(ref, str):
                    try:
                        matched_node = nodes_by_id.get(UUID(ref))
                    except ValueError:
                        matched_node = None
                if matched_node is None:
                    unmatched_refs.append(_format_unmatched_highlight(ref))
                else:
                    matched_nodes.add(matched_node)
            return matched_nodes, unmatched_refs

        unmatched_highlight_refs: list[str] = []
        if highlight_nodes is None:
            highlight_groups: list[tuple[set[TannerNode], str, str | None]] = []
        elif isinstance(highlight_nodes, set):
            matched_nodes, unmatched_refs = _resolve_highlight_nodes(
                cast(HighlightNodeSet, highlight_nodes)
            )
            unmatched_highlight_refs.extend(unmatched_refs)
            highlight_groups = [
                (
                    matched_nodes,
                    "gold",
                    None,
                )
            ]
        else:
            highlight_groups = []
            for highlight_group in highlight_nodes:
                if len(highlight_group) == 2:
                    nodes, color = highlight_group
                    label = None
                else:
                    nodes, color, label = highlight_group
                color = _resolve_highlight_color(color)
                matched_nodes, unmatched_refs = _resolve_highlight_nodes(nodes)
                unmatched_highlight_refs.extend(unmatched_refs)
                if matched_nodes:
                    highlight_groups.append((matched_nodes, color, label))

        node_highlight_colors: dict[TannerNode, str] = {}
        highlight_legend_labels: dict[str, str] = {}
        for nodes, color, label in highlight_groups:
            for node in nodes:
                node_highlight_colors[node] = color
            if label is not None:
                highlight_legend_labels[color] = label
        highlighted_nodes = set(node_highlight_colors)

        coords_by_node = {
            node: cast(
                tuple[int, ...],
                node.coordinates if node.coordinates is not None else tuple(),
            )
            for node in all_nodes
        }
        drawable_edges = [
            edge
            for edge in graph.edges
            if edge.variable_node in coords_by_node
            and edge.check_node in coords_by_node
        ]

        coord_lengths = {len(coords_by_node[node]) for node in all_nodes}
        if 0 in coord_lengths and len(coord_lengths) > 1:
            raise ValueError("Either all nodes must have coordinates or none.")
        if len(coord_lengths) > 1:
            raise ValueError("All node coordinates must have the same dimension.")

        dim = coord_lengths.pop()
        if dim not in {0, 2, 3, 4}:
            raise ValueError("Node coordinates must be empty, 2D, 3D, or 4D.")

        def _system_title(system: tuple[int, int]) -> str:
            if system_labels is None:
                return f"System {system}"
            if callable(system_labels):
                custom_label = system_labels(system)
            else:
                custom_label = f"{system_labels.get(system)} - {system}"
            return custom_label if custom_label is not None else f"System {system}"

        def _draw_nodes(ax, nodes, pos, marker, size, highlighted_size, color):
            regular_nodes = [node for node in nodes if node not in highlighted_nodes]
            if regular_nodes:
                ax.scatter(
                    [pos[node][0] for node in regular_nodes],
                    [pos[node][1] for node in regular_nodes],
                    marker=marker,
                    s=size,
                    color=color,
                    zorder=3,
                )

            nodes_by_highlight_color: dict[str, list[TannerNode]] = {}
            for node in nodes:
                highlight_color = node_highlight_colors.get(node)
                if highlight_color is not None:
                    nodes_by_highlight_color.setdefault(highlight_color, []).append(
                        node
                    )

            for highlight_color, colored_nodes in nodes_by_highlight_color.items():
                ax.scatter(
                    [pos[node][0] for node in colored_nodes],
                    [pos[node][1] for node in colored_nodes],
                    marker=marker,
                    s=highlighted_size,
                    color=color,
                    edgecolors=highlight_color,
                    linewidths=2.2,
                    zorder=4,
                )

        def _annotate_nodes(ax, nodes, pos):
            for node in nodes:
                x, y = pos[node]
                ax.text(x + 0.03, y + 0.03, node.tag, fontsize=8)

        def _build_legend_items():
            legend_items = [
                Line2D(
                    [0], [0], marker="o", linestyle="", color="black", label="Variable"
                ),
                Line2D(
                    [0], [0], marker="s", linestyle="", color="dimgray", label="Check"
                ),
                Line2D([0], [0], color="tab:red", label="X edge"),
                Line2D([0], [0], color="tab:blue", label="Z edge"),
                Line2D([0], [0], color="tab:purple", label="Y edge"),
            ]
            for color in dict.fromkeys(node_highlight_colors.values()):
                legend_items.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markeredgewidth=2.2,
                        color=color,
                        label=highlight_legend_labels.get(
                            color, f"Highlighted ({color})"
                        ),
                    )
                )
            return legend_items

        def _build_unmatched_legend_items():
            unmatched_items = []
            for unmatched_ref in dict.fromkeys(unmatched_highlight_refs):
                unmatched_items.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="",
                        color="none",
                        label=unmatched_ref,
                    )
                )
            return unmatched_items

        def _add_unmatched_legend(ax, main_legend):
            unmatched_items = _build_unmatched_legend_items()
            if not unmatched_items:
                return
            ax.legend(
                handles=unmatched_items,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                title="Unmatched highlights",
                handlelength=0,
                handletextpad=0,
            )
            if main_legend is not None:
                ax.add_artist(main_legend)

        def _add_shared_legend(fig, axes):
            legend_items = _build_legend_items()
            target_ax = axes[0]
            main_legend = None
            if legend_items:
                main_legend = target_ax.legend(
                    handles=legend_items, loc="best", fontsize=8
                )
            _add_unmatched_legend(target_ax, main_legend)

        def _draw_single_axis(ax, pos):
            for edge in drawable_edges:
                x1, y1 = pos[edge.variable_node]
                x2, y2 = pos[edge.check_node]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_color.get(edge.pauli_checked, "gray"),
                    linewidth=1.6,
                    alpha=0.85,
                    zorder=1,
                )

            var_nodes = sorted(graph.variable_nodes, key=lambda n: n.tag)
            check_nodes = sorted(graph.check_nodes, key=lambda n: n.tag)

            _draw_nodes(
                ax,
                var_nodes,
                pos,
                marker="o",
                size=120,
                highlighted_size=180,
                color="black",
            )
            _draw_nodes(
                ax,
                check_nodes,
                pos,
                marker="s",
                size=130,
                highlighted_size=190,
                color="dimgray",
            )
            _annotate_nodes(ax, var_nodes + check_nodes, pos)

            main_legend = ax.legend(
                handles=_build_legend_items(), loc="best", fontsize=8
            )
            _add_unmatched_legend(ax, main_legend)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")

        if dim == 0:
            fig, ax = plt.subplots(figsize=(9, 5))

            var_nodes = sorted(graph.variable_nodes, key=lambda n: n.tag)
            check_nodes = sorted(graph.check_nodes, key=lambda n: n.tag)
            css_like = all(
                c.check_type in {PauliChar.X, PauliChar.Z} for c in check_nodes
            )

            pos = {}
            for i, node in enumerate(var_nodes):
                pos[node] = (0.0, float(-i))

            if css_like:
                x_checks = [c for c in check_nodes if c.check_type == PauliChar.X]
                z_checks = [c for c in check_nodes if c.check_type == PauliChar.Z]
                for i, node in enumerate(sorted(x_checks, key=lambda n: n.tag)):
                    pos[node] = (-1.0, float(-i))
                for i, node in enumerate(sorted(z_checks, key=lambda n: n.tag)):
                    pos[node] = (1.0, float(-i))
            else:
                for i, node in enumerate(check_nodes):
                    pos[node] = (1.0, float(-i))

            _draw_single_axis(ax, pos)
            ax.set_title("Tanner Graph (Bipartite Layout)")
            return _finalize_figure(fig, ax)

        if dim == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            pos = {
                node: (coords_by_node[node][0], coords_by_node[node][1])
                for node in all_nodes
            }
            if not periodic:
                _draw_single_axis(ax, pos)
                ax.set_title("Tanner Graph (2D Coordinates)")
                return _finalize_figure(fig, ax)

            # Periodic rendering for toroidal layouts: wrap-across edges are drawn
            # as two boundary-touching segments so periodic connectivity is explicit.
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = (max_x - min_x) + 1
            span_y = (max_y - min_y) + 1
            pad = 0.45

            for edge in drawable_edges:
                x1, y1 = pos[edge.variable_node]
                x2, y2 = pos[edge.check_node]
                color = edge_color.get(edge.pauli_checked, "gray")

                dx = x2 - x1
                dy = y2 - y1
                wraps_x = abs(dx) > (span_x / 2)
                wraps_y = abs(dy) > (span_y / 2)

                if not wraps_x and not wraps_y:
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=1.6,
                        alpha=0.85,
                        zorder=1,
                    )
                    continue

                if wraps_x and not wraps_y:
                    if dx > 0:
                        ax.plot(
                            [x1, min_x - pad],
                            [y1, y1],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [max_x + pad, x2],
                            [y2, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    else:
                        ax.plot(
                            [x1, max_x + pad],
                            [y1, y1],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [min_x - pad, x2],
                            [y2, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    continue

                if wraps_y and not wraps_x:
                    if dy > 0:
                        ax.plot(
                            [x1, x1],
                            [y1, min_y - pad],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [x2, x2],
                            [max_y + pad, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    else:
                        ax.plot(
                            [x1, x1],
                            [y1, max_y + pad],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [x2, x2],
                            [min_y - pad, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    continue

                # Fallback for rare diagonal wrap cases.
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    linewidth=1.2,
                    alpha=0.6,
                    zorder=1,
                    linestyle="--",
                )

            var_nodes = sorted(graph.variable_nodes, key=lambda n: n.tag)
            check_nodes = sorted(graph.check_nodes, key=lambda n: n.tag)

            _draw_nodes(
                ax,
                var_nodes,
                pos,
                marker="o",
                size=120,
                highlighted_size=180,
                color="black",
            )
            _draw_nodes(
                ax,
                check_nodes,
                pos,
                marker="s",
                size=130,
                highlighted_size=190,
                color="dimgray",
            )
            _annotate_nodes(ax, var_nodes + check_nodes, pos)

            main_legend = ax.legend(
                handles=_build_legend_items(), loc="best", fontsize=8
            )
            _add_unmatched_legend(ax, main_legend)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")
            ax.set_xlim(min_x - (pad + 0.15), max_x + (pad + 0.15))
            ax.set_ylim(min_y - (pad + 0.15), max_y + (pad + 0.15))
            ax.set_title("Tanner Graph (2D Toroidal Coordinates)")
            return _finalize_figure(fig, ax)

        if dim == 3:
            planes = sorted({coords_by_node[node][2] for node in all_nodes})
            n_rows = 2
            n_cols = (len(planes) + n_rows - 1) // n_rows
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6 * n_cols, 5 * n_rows),
                squeeze=False,
            )
            axes_flat = list(axes.ravel())
            axes_list = axes_flat[: len(planes)]
            for ax in axes_flat[len(planes) :]:
                ax.set_axis_off()
            plane_to_ax = {plane: axes_list[i] for i, plane in enumerate(planes)}
            plane_to_row = {plane: i // n_cols for i, plane in enumerate(planes)}

            pos_2d = {
                node: (coords_by_node[node][0], coords_by_node[node][1])
                for node in all_nodes
            }
            cross_plane_edges = []
            for edge in drawable_edges:
                p_var = coords_by_node[edge.variable_node][2]
                p_chk = coords_by_node[edge.check_node][2]
                if p_var == p_chk:
                    ax = plane_to_ax[p_var]
                    x1, y1 = pos_2d[edge.variable_node]
                    x2, y2 = pos_2d[edge.check_node]
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=edge_color.get(edge.pauli_checked, "gray"),
                        linewidth=1.6,
                        alpha=0.85,
                        zorder=1,
                    )
                else:
                    cross_plane_edges.append(edge)

            for plane, ax in plane_to_ax.items():
                plane_var = sorted(
                    [n for n in graph.variable_nodes if coords_by_node[n][2] == plane],
                    key=lambda n: n.tag,
                )
                plane_check = sorted(
                    [n for n in graph.check_nodes if coords_by_node[n][2] == plane],
                    key=lambda n: n.tag,
                )

                _draw_nodes(
                    ax,
                    plane_var,
                    pos_2d,
                    marker="o",
                    size=120,
                    highlighted_size=180,
                    color="black",
                )
                _draw_nodes(
                    ax,
                    plane_check,
                    pos_2d,
                    marker="s",
                    size=130,
                    highlighted_size=190,
                    color="dimgray",
                )

                _annotate_nodes(ax, plane_var + plane_check, pos_2d)

                ax.set_title(f"Plane {plane}")
                ax.set_aspect("equal", adjustable="datalim")
                if invert_y_rows and plane_to_row[plane] in invert_y_rows:
                    ax.invert_yaxis()
                ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")

            for edge in cross_plane_edges:
                x1, y1 = pos_2d[edge.variable_node]
                x2, y2 = pos_2d[edge.check_node]
                a1 = plane_to_ax[coords_by_node[edge.variable_node][2]]
                a2 = plane_to_ax[coords_by_node[edge.check_node][2]]
                connector = ConnectionPatch(
                    xyA=(x1, y1),
                    xyB=(x2, y2),
                    coordsA="data",
                    coordsB="data",
                    axesA=a1,
                    axesB=a2,
                    color=edge_color.get(edge.pauli_checked, "gray"),
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.7,
                )
                fig.add_artist(connector)

            _add_shared_legend(fig, axes_list)
            fig.suptitle("Tanner Graph (3D Coordinates by Plane)")
            fig.tight_layout()
            return _finalize_figure(fig, axes_list)

        systems = sorted(
            {(coords_by_node[node][2], coords_by_node[node][3]) for node in all_nodes}
        )
        if any(sys_x < 0 or sys_y < 0 for sys_x, sys_y in systems):
            raise ValueError(
                "For 4D coordinates, system coordinates must be non-negative integers."
            )
        n_cols = max(sys_x for sys_x, _ in systems) + 1
        n_rows = max(sys_y for _, sys_y in systems) + 1

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )

        system_to_display_row = {system: n_rows - 1 - system[1] for system in systems}
        system_to_ax = {
            system: axes[system_to_display_row[system], system[0]] for system in systems
        }
        occupied_slots = {
            (system_to_display_row[system], system[0]) for system in systems
        }
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if (row_idx, col_idx) not in occupied_slots:
                    axes[row_idx, col_idx].set_axis_off()

        axes_list = [system_to_ax[system] for system in systems]

        pos_2d = {
            node: (coords_by_node[node][0], coords_by_node[node][1])
            for node in all_nodes
        }

        cross_system_edges = []
        for edge in drawable_edges:
            s_var = (
                coords_by_node[edge.variable_node][2],
                coords_by_node[edge.variable_node][3],
            )
            s_chk = (
                coords_by_node[edge.check_node][2],
                coords_by_node[edge.check_node][3],
            )
            if s_var == s_chk:
                ax = system_to_ax[s_var]
                x1, y1 = pos_2d[edge.variable_node]
                x2, y2 = pos_2d[edge.check_node]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_color.get(edge.pauli_checked, "gray"),
                    linewidth=1.6,
                    alpha=0.85,
                    zorder=1,
                )
            else:
                cross_system_edges.append(edge)

        for system, ax in system_to_ax.items():
            system_var = sorted(
                [
                    n
                    for n in graph.variable_nodes
                    if (coords_by_node[n][2], coords_by_node[n][3]) == system
                ],
                key=lambda n: n.tag,
            )
            system_check = sorted(
                [
                    n
                    for n in graph.check_nodes
                    if (coords_by_node[n][2], coords_by_node[n][3]) == system
                ],
                key=lambda n: n.tag,
            )

            _draw_nodes(
                ax,
                system_var,
                pos_2d,
                marker="o",
                size=120,
                highlighted_size=180,
                color="black",
            )
            _draw_nodes(
                ax,
                system_check,
                pos_2d,
                marker="s",
                size=130,
                highlighted_size=190,
                color="dimgray",
            )

            _annotate_nodes(ax, system_var + system_check, pos_2d)

            ax.set_title(_system_title(system))
            ax.set_aspect("equal", adjustable="datalim")
            if invert_y_rows and system_to_display_row[system] in invert_y_rows:
                ax.invert_yaxis()
            ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")

        for edge in cross_system_edges:
            x1, y1 = pos_2d[edge.variable_node]
            x2, y2 = pos_2d[edge.check_node]
            s1 = (
                coords_by_node[edge.variable_node][2],
                coords_by_node[edge.variable_node][3],
            )
            s2 = (
                coords_by_node[edge.check_node][2],
                coords_by_node[edge.check_node][3],
            )
            a1 = system_to_ax[s1]
            a2 = system_to_ax[s2]
            connector = ConnectionPatch(
                xyA=(x1, y1),
                xyB=(x2, y2),
                coordsA="data",
                coordsB="data",
                axesA=a1,
                axesB=a2,
                color=edge_color.get(edge.pauli_checked, "gray"),
                linestyle="--",
                linewidth=1.1,
                alpha=0.7,
            )
            fig.add_artist(connector)

        _add_shared_legend(fig, axes_list)
        fig.suptitle(
            _plot_title(
                title
                if title is not None
                else "Tanner Graph (4D Coordinates by System)"
            )
        )
        fig.tight_layout()
        return _finalize_figure(fig, axes_list)
