"""
Forensic Data Module
SPEC-FORENSIC-001 Phase 2-A: Crime Language Pattern Database

Contains JSON pattern databases for:
- Gaslighting patterns (가스라이팅 패턴)
- Threat patterns (협박 패턴)
- Coercion patterns (강압 패턴)
- Deception markers (기만 언어 지표)
"""

import json
from pathlib import Path
from typing import Dict, Any

DATA_DIR = Path(__file__).parent


def load_json_db(filename: str) -> Dict[str, Any]:
    """Load a JSON database file from the forensic data directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"JSON database not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gaslighting_patterns() -> Dict[str, Any]:
    """Load gaslighting patterns database."""
    return load_json_db("gaslighting_patterns_ko.json")


def get_threat_patterns() -> Dict[str, Any]:
    """Load threat patterns database."""
    return load_json_db("threat_patterns_ko.json")


def get_coercion_patterns() -> Dict[str, Any]:
    """Load coercion patterns database."""
    return load_json_db("coercion_patterns_ko.json")


def get_deception_markers() -> Dict[str, Any]:
    """Load deception markers database."""
    return load_json_db("deception_markers_ko.json")
