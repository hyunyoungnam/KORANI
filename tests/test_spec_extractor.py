"""Spec Extractor tests — offline with stub LLM."""

import json

import pytest

from korani.agents.spec_extractor import SpecExtractionError, SpecExtractor
from korani.models import TaskSpec

TASK = TaskSpec(mode="A", domain="battery", task_en="reproduce degradation model")

CANNED = {
    "title": "Lithium-ion battery degradation: how to model it",
    "domain": "battery",
    "solver": "pybamm",
    "model_summary": "DFN (P2D) with SEI growth and lithium plating submodels",
    "governing_equations": ["DFN electrochemistry", "SEI growth kinetics"],
    "geometry": "1D cell sandwich, LG M50 21700 parameters",
    "materials": ["graphite-SiOx anode", "NMC811 cathode"],
    "parameters": [
        {"name": "SEI kinetic rate constant", "symbol": "k_SEI",
         "value": "1e-12", "units": "m s^-1", "source": "paper", "notes": ""},
        {"name": "plating exchange current density", "symbol": "j0_pl",
         "value": None, "units": "A m^-2", "source": "missing",
         "notes": "only shown in a plot"},
    ],
    "operating_conditions": ["1C cycling at 25degC, 4.2V-2.5V window"],
    "numerical_settings": [],
    "target_results": [
        {"description": "capacity fade over 1000 cycles", "location": "Figure 5",
         "quantity": "discharge capacity", "value": None}
    ],
    "ambiguities": [
        {"field": "parameters.j0_pl", "issue": "value only shown in a plot",
         "candidates": ["digitize Figure 2", "use PyBaMM default"]}
    ],
}


class StubClient:
    def __init__(self, response):
        self.response = response

    def chat(self, model, messages, temperature=0.2, max_tokens=1024):
        return self.response


def test_full_spec_parses():
    spec = SpecExtractor(StubClient(json.dumps(CANNED)), model="stub").extract(TASK, "text")
    assert spec.solver == "pybamm"
    assert spec.parameters[1].source == "missing"
    assert spec.parameters[1].value is None          # honesty rule: not invented
    assert spec.ambiguities[0].candidates            # feeds branch-on-ambiguity
    assert spec.target_results[0].location == "Figure 5"
    assert spec.work_id is None                      # set only on persistence


def test_bad_enums_normalized():
    payload = dict(CANNED)
    payload["domain"] = "chemistry"
    payload["solver"] = "comsol"
    spec = SpecExtractor(StubClient(json.dumps(payload)), model="stub").extract(TASK, "t")
    assert spec.domain == "unknown"
    assert spec.solver == "none"


def test_llm_cannot_claim_work_id():
    payload = dict(CANNED)
    payload["work_id"] = "fake-id"
    spec = SpecExtractor(StubClient(json.dumps(payload)), model="stub").extract(TASK, "t")
    assert spec.work_id is None


def test_fenced_output_accepted():
    fenced = "```json\n" + json.dumps(CANNED) + "\n```"
    spec = SpecExtractor(StubClient(fenced), model="stub").extract(TASK, "t")
    assert spec.title.startswith("Lithium-ion")


def test_garbage_raises_with_raw():
    with pytest.raises(SpecExtractionError) as excinfo:
        SpecExtractor(StubClient("cannot comply"), model="stub").extract(TASK, "t")
    assert excinfo.value.raw_output == "cannot comply"
