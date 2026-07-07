"""PyBaMM code-generation template context."""

from __future__ import annotations

from korani.knowledge.base import KnowledgeModule, TemplateContext
from korani.models import EvaluationContract, SimulationSpec


class PybammKnowledgeModule(KnowledgeModule):
    solver = "pybamm"

    def build_template_context(
        self, spec: SimulationSpec, contract: EvaluationContract
    ) -> TemplateContext:
        curve_keys = [c.key for c in contract.checks if c.kind != "numeric"]
        numeric_keys = [
            c.key for c in contract.checks if c.kind == "numeric" and c.expected_value is not None
        ]
        return TemplateContext(
            name="PyBaMM battery scaffold",
            instructions=(
                "Use this as a structural scaffold for PyBaMM scripts. Choose the "
                "simplest PyBaMM model class that matches the SimulationSpec. Do not "
                "invent parameter-set names or output variables; if a name is uncertain, "
                "use a conservative built-in default and mark it with '# ASSUMPTION:'. "
                f"The script must write {contract.results_file!r} and curve CSV files "
                f"for: {curve_keys}. Numeric result keys, if any: {numeric_keys}."
            ),
            template=r'''
import csv
import json

import pybamm


RESULTS_FILE = "results.json"


def write_curve(path: str, x_name: str, y_name: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([x_name, y_name])
        writer.writerows(rows)


def build_simulation():
    """Create the PyBaMM model, parameters, experiment, and Simulation."""
    # ASSUMPTION: replace with SPM/SPMe/DFN according to the SimulationSpec.
    model = pybamm.lithium_ion.DFN()

    # ASSUMPTION: replace with a paper-specified or suitable built-in set.
    parameter_values = pybamm.ParameterValues("Chen2020")

    simulation = pybamm.Simulation(model, parameter_values=parameter_values)
    return simulation


def extract_results(solution):
    """Return (results_dict, curves_dict) using solution variables."""
    results = {}
    curves = {}

    # Example:
    # voltage = solution["Terminal voltage [V]"]
    # rows = list(zip(voltage.entries[:, 0].tolist(), voltage.data[:, 0].tolist()))
    # curves["voltage_curve"] = rows
    return results, curves


def main():
    simulation = build_simulation()
    solution = simulation.solve()
    results, curves = extract_results(solution)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for key, rows in curves.items():
        write_curve(f"{key}.csv", "x", "y", rows)


if __name__ == "__main__":
    main()
'''.strip(),
        )
