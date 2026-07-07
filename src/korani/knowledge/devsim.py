"""DEVSIM code-generation template context."""

from __future__ import annotations

from korani.knowledge.base import KnowledgeModule, TemplateContext
from korani.models import EvaluationContract, SimulationSpec


class DevsimKnowledgeModule(KnowledgeModule):
    solver = "devsim"

    def build_template_context(
        self, spec: SimulationSpec, contract: EvaluationContract
    ) -> TemplateContext:
        curve_keys = [c.key for c in contract.checks if c.kind != "numeric"]
        numeric_keys = [
            c.key for c in contract.checks if c.kind == "numeric" and c.expected_value is not None
        ]
        return TemplateContext(
            name="DEVSIM TCAD scaffold",
            instructions=(
                "Use this as a structural scaffold for DEVSIM scripts. Do not invent "
                "nonexistent DEVSIM APIs or pretend an external mesh file exists. If a "
                "mesh file is required, check it with pathlib.Path before loading it; "
                "otherwise create the mesh in code using real DEVSIM mesh APIs. Keep "
                "all assumed values marked with '# ASSUMPTION:'. The script must write "
                f"{contract.results_file!r} and curve CSV files for: {curve_keys}. "
                f"Numeric result keys, if any: {numeric_keys}."
            ),
            template=r'''
import csv
import json
from pathlib import Path

import devsim


RESULTS_FILE = "results.json"


def require_api(name: str):
    if not hasattr(devsim, name):
        raise RuntimeError(f"Installed devsim module has no API: devsim.{name}")
    return getattr(devsim, name)


def write_curve(path: str, x_name: str, y_name: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([x_name, y_name])
        writer.writerows(rows)


def build_device():
    """Create/load mesh, regions, contacts, material parameters, and models.

    Fill this function using real DEVSIM APIs only. If the paper does not
    provide enough geometry/mesh information, build the simplest honest
    approximating device and mark every choice with '# ASSUMPTION:'.
    """
    # Example guard for external mesh use:
    # mesh_path = Path("device.msh")
    # if not mesh_path.exists():
    #     raise FileNotFoundError("Required mesh file is missing: device.msh")
    #
    # TODO: create mesh/regions/contacts and define Poisson/continuity models.
    raise NotImplementedError("DEVSIM device setup must be filled from the spec.")


def run_sweeps():
    """Run bias sweeps and return (results_dict, curves_dict)."""
    # TODO: call devsim.solve(...) over the required operating conditions.
    # Curves dict format:
    # {"check_key": [("bias", "current"), ...]}
    return {}, {}


def main():
    build_device()
    results, curves = run_sweeps()

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for key, rows in curves.items():
        write_curve(f"{key}.csv", "x", "y", rows)


if __name__ == "__main__":
    main()
'''.strip(),
        )
