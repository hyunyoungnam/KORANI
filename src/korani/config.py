"""Configuration loading: config.yaml + environment overrides.

Precedence (KoCoScientist pattern): env var > config.yaml > default.
Env vars: KORANI_LLM_BASE_URL, KORANI_LLM_API_KEY, KORANI_MODEL_INTERPRETER.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULTS: Dict[str, Any] = {
    "llm": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "not-needed",
    },
    "data_dir": "data",
    "models": {
        "interpreter": "koni",
        "search_planner": "koni",
        "paper_triage": "koni",
        "spec_extractor": "koni",
        "evaluator": "koni",
        "engineer": "koni",
        "debugger": "koni",
        "result_analyst": "koni",
        "proposer_critic": "koni",
    },
    "extraction": {
        "max_chars": 24000,  # paper text budget per Spec Extractor LLM call
        "chunk_overlap": 2000,  # overlap between chunks (two-pass extraction)
        "max_chunks": 6,  # cost cap on chunked extraction
    },
    "generation": {
        "temperature": 0.2,
        "max_tokens": 1024,
    },
    "execution": {
        "timeout_seconds": 900,  # wall-clock limit per solver run (stage E)
    },
    "budget": {
        "max_solver_runs": 6,  # per-task cap: stage E variants + retries + stage F rungs
        "max_debug_retries": 2,  # Debugger attempts per variant
        "max_variants": 2,  # branch-on-ambiguity fan-out (NOT a tree)
    },
    "analysis": {
        "use_vision": True,  # attach plot/page images for the Result Analyst
        "max_curve_points": 40,  # CSV rows shown to the analyst as text
    },
    "search": {
        "providers": ["openalex", "semanticscholar"],
        "per_query_limit": 10,
        "max_candidates": 12,
        "shortlist_size": 5,
        "mailto": "",  # set your email → OpenAlex polite pool (faster, stabler)
        # Behind corporate SSL inspection: prefer SSL_CERT_FILE=<corp CA .pem>;
        # verify_ssl=false is the last-resort escape hatch for testing.
        "verify_ssl": True,
    },
}


def _find_config_file(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    # cwd, then project root (src/korani/config.py → 3 levels up)
    candidates = [
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent.parent.parent / "config.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config = dict(_DEFAULTS)
    config_file = _find_config_file(path)
    if config_file is not None:
        with open(config_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        config = _deep_merge(config, loaded)

    # Environment overrides
    env_base_url = os.getenv("KORANI_LLM_BASE_URL")
    if env_base_url:
        config["llm"]["base_url"] = env_base_url
    env_api_key = os.getenv("KORANI_LLM_API_KEY")
    if env_api_key:
        config["llm"]["api_key"] = env_api_key
    # Per-role model override: KORANI_MODEL_<ROLE>, e.g. KORANI_MODEL_PAPER_TRIAGE
    for role in list(config["models"].keys()):
        env_model = os.getenv(f"KORANI_MODEL_{role.upper()}")
        if env_model:
            config["models"][role] = env_model

    return config
