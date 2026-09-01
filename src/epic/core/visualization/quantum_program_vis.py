from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from epic.core.data_structure.physical_qubit import PhysicalQubit
from epic.core.data_structure.quantum_program import QuantumProgram


class QuantumProgramVisualizer:
	@staticmethod
	def visualize(
		program: QuantumProgram,
		output_path: str | Path | None = None,
		title: str | None = None,
	):
		"""Render a quantum program as a time-versus-qubit schedule.

		Positive-length operations are rendered as boxes whose widths match
		their durations. Zero-length operations are rendered as vertical tick
		marks. A small inset separates neighboring operations visually.

		Parameters
		----------
		program : QuantumProgram
			Quantum program to render.
		output_path : str | Path | None, optional
			Save the figure to this path instead of returning it.
		title : str | None, optional
			Override the default figure title.

		Returns
		-------
		tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | None
			The created figure and axes, or ``None`` when saved to a file.
		"""

		if all(isinstance(qubit, PhysicalQubit) for qubit in program.qubits):
			qubits = sorted(program.qubits, key=lambda qubit: qubit.integer_index)
		else:
			qubits = program.qubits
		lane_by_qubit = {qubit: index for index, qubit in enumerate(qubits)}
		box_inset = 0.12
		cnot_indices_by_slot = {}
		for operation_index, (operation, start) in enumerate(program.operations):
			if operation.name in {"CNOT", "CX"} and len(operation.targets) == 2:
				cnot_indices_by_slot.setdefault((start, operation.length), []).append(
					operation_index
				)

		display_start_by_operation = {}
		cnot_display_start_by_operation = {}
		extra_time = 0
		for start in sorted({start for _, start in program.operations}):
			slot_operations = [
				(operation_index, operation)
				for operation_index, (operation, operation_start) in enumerate(
					program.operations
				)
				if operation_start == start
			]
			for operation_index, _ in slot_operations:
				display_start_by_operation[operation_index] = start + extra_time

			for (slot_start, length), operation_indices in cnot_indices_by_slot.items():
				if slot_start != start:
					continue
				for index, operation_index in enumerate(operation_indices):
					cnot_display_start_by_operation[operation_index] = (
						start + extra_time + index * length
					)
				extra_time += (len(operation_indices) - 1) * length

		operation_display_start = {
			operation_index: cnot_display_start_by_operation.get(
				operation_index, display_start
			)
			for operation_index, display_start in display_start_by_operation.items()
		}
		visual_depth = max(
			[
				operation_display_start[operation_index] + operation.length
				for operation_index, (operation, _) in enumerate(program.operations)
			]
			or [1]
		)

		fig_height = max(3.0, 0.75 * len(qubits) + 1.5)
		fig_width = max(10.0, 0.6 * visual_depth + 2.0)
		fig, ax = plt.subplots(figsize=(fig_width, fig_height))
		ax.set_title(title if title is not None else program.name)
		ax.set_xlabel("Time")
		ax.set_ylabel("Qubit")
		ax.set_axisbelow(True)

		for lane in range(len(qubits)):
			ax.axhline(lane, color="black", linewidth=0.8, alpha=0.55, zorder=1)

		for operation_index, (operation, start) in enumerate(program.operations):
			target_lanes = sorted(lane_by_qubit[qubit] for qubit in operation.targets)
			if not target_lanes:
				continue
			display_start = operation_display_start[operation_index]

			if operation.length == 0:
				ax.vlines(
					display_start,
					target_lanes[0] - 0.35,
					target_lanes[-1] + 0.35,
					color="black",
					linewidth=2.0,
					zorder=4,
				)
				ax.text(
					display_start + 0.05,
					target_lanes[-1] + 0.4,
					operation.name,
					fontsize=8,
					va="bottom",
				)
				continue

			for lane in target_lanes:
				box_width = max(operation.length - 2 * box_inset, 0.2)
				ax.add_patch(
					Rectangle(
						(display_start + box_inset, lane - 0.3),
						box_width,
						0.6,
						facecolor="lightsteelblue",
						edgecolor="black",
						linewidth=1.0,
						zorder=3,
					)
				)
				ax.text(
					display_start + operation.length / 2,
					lane,
					operation.name,
					ha="center",
					va="center",
					fontsize=8,
					zorder=4,
				)

			if len(target_lanes) > 1:
				ax.vlines(
					display_start + operation.length / 2,
					target_lanes[0],
					target_lanes[-1],
					color="black",
					linewidth=1.2,
					zorder=2,
				)

		ax.set_yticks(range(len(qubits)))
		ax.set_yticklabels([qubit.integer_index if isinstance(qubit, PhysicalQubit) else qubit.name for qubit in qubits])
		ax.set_ylim(-0.6, max(len(qubits) - 1, 0) + 0.6)
		ax.set_xlim(-0.5, max(visual_depth, 1) + 0.5)
		ax.set_xticks([])
		ax.grid(axis="x", alpha=0.2, linestyle=":")
		ax.spines["left"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.spines["top"].set_visible(False)

		fig.tight_layout()

		if output_path is not None:
			path = Path(output_path)
			path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(path, bbox_inches="tight")
			plt.close(fig)
			return None

		return fig, ax


def draw_quantum_program(
	program: QuantumProgram,
	output_path: str | Path | None = None,
	title: str | None = None,
):
	"""Draw ``program`` using :class:`QuantumProgramVisualizer`."""

	return QuantumProgramVisualizer.visualize(
		program,
		output_path=output_path,
		title=title,
	)
