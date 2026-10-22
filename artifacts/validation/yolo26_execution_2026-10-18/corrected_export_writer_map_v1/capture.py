"""Retain the current source identities of a read-only writer/API mapping."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FILES = (
    'src/scoring/corrected_rule_rebuild.py',
    'src/scoring/corrected_shot_rebuild.py',
    'src/scoring/corrected_forecasts.py',
    'src/scoring/point_timeline.py',
    'src/utils/point_analysis_sequence.py',
    'src/scoring/scoring_engine.py',
    'src/utils/scoring_journal.py',
    'src/scoring/rule_replay.py',
    'src/utils/analytics_json_artifact.py',
    'src/utils/json_array_artifact.py',
    'src/utils/point_timeline_artifact.py',
    'src/pipeline/phase6_pipeline.py',
    'src/visualization/landing_zones.py',
    'src/utils/landing_zone_sequence.py',
    'src/api/analysis_resume.py',
    'src/api/analysis_generation.py',
    'src/api/result_document.py',
    'src/api/results.py',
    'src/api/point_timeline_contract.py',
    'src/api/worker.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = {name: sha(ROOT / name) for name in FILES}
for name in FILES:
    target = HERE / 'source_snapshot' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)
after = {name: sha(ROOT / name) for name in FILES}
assert before == after
record = {
    'scope': 'READ_ONLY_CURRENT_NATIVE_WRITER_AND_API_MAPPING; NO_EXPORT_OR_RENDER_OR_PUBLICATION_EXECUTION',
    'captured_at_utc': datetime.now(timezone.utc).isoformat(),
    'mapped_source_file_count': len(FILES),
    'source_sha256': before,
    'mapping_sha256': sha(HERE / 'MAPPING.md'),
    'source_pins_stable_during_capture': before == after,
    'implementation_status': 'NOT_IMPLEMENTED_BY_THIS_MAPPING',
    'actual_sql_index_receipt_sha256': sha(HERE.parent / 'prediction_sql_index_checks_v1' / 'receipt.json'),
    'no_application_source_or_documentation_mutations': True,
}
(HERE / 'source_pins.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record, indent=2))
