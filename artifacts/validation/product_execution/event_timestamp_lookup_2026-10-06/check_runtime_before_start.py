"""Read the reviewed runtime topology without starting/stopping any process."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

unit = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('reviewed_event_runtime', unit / 'verify_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report = {'observedAt': datetime.now(timezone.utc).isoformat(), 'readOnly': True,
          'runtime': module.runtime_processes(), 'snapshot': module.snapshot(),
          'sourceBindings': module.source_bindings()}
with (unit / 'runtime_before_start.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps({'state': report['runtime']['state'], 'savedRows': len(report['snapshot']['jobs']),
                  'savedFiles': len(report['snapshot']['savedAnalysisFiles'])}))
