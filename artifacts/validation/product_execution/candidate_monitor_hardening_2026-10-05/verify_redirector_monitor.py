"""Run the retained small guard protocol against the hardened monitor once."""
import hashlib
import importlib.util
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'artifacts/validation/product_execution/candidate_storage_measurement_2026-10-05/verify_redirector_monitor.py'
spec = importlib.util.spec_from_file_location('retained_small_monitor_protocol', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OWN = OWN
module.main()
with (OWN / 'launcher_binding.json').open('x', encoding='utf-8') as stream:
    json.dump({'launcherSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'retainedProtocolSha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'scope': 'Same controlled 64 MiB guard proof after teardown hardening; no full benchmark or model rerun.'}, stream, indent=2)
