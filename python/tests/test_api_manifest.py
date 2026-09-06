"""Prevent removal of the reviewed Python API and trade serialization coverage."""

import json
import runpy
from pathlib import Path


def test_reviewed_rust_api_is_exposed():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "api_manifest.json").read_text())
    checker = runpy.run_path(str(root / "scripts" / "check_api.py"))
    failures = checker["check_python"](manifest)
    assert not failures, "\n".join(failures)
    paths = [record["rust"] for record in manifest["items"]]
    assert len(paths) == len(set(paths))


def test_every_native_trade_variant_has_a_roundtrip_fixture():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "api_manifest.json").read_text())
    record = next(
        record for record in manifest["items"] if record["rust"] == "openferric::instruments::TradeInstrument"
    )
    fixtures = json.loads((root / "tests" / "data" / "trade_instruments.json").read_text())
    assert {fixture["type"] for fixture in fixtures} == set(record["variants"])
