"""Fresh registration-storage compatibility using the retained guarded smoke."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RETAINED = ROOT / 'scripts/audit/verify_candidate_storage_pipeline_smoke.py'
spec = importlib.util.spec_from_file_location('retained_registration_compatibility_guard', RETAINED)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
smoke.UNIT = ROOT / 'artifacts/validation/product_execution/registration_storage_real_smoke_2026-10-05'
smoke.RUN = smoke.UNIT / 'run'
smoke.SOURCE_BINDINGS += ('src/utils/frame_registration_sequence.py',
                          'scripts/audit/verify_registration_storage_pipeline_smoke.py')
# The retained guard dispatches its own __file__ to the clean child. Point that
# dispatch at this wrapper so both parent and child use the same fresh paths.
smoke.__file__ = __file__
retained_compare_artifacts = smoke.compare_artifacts


def compare_registration_artifacts(helper):
    scratch = [path.relative_to(smoke.RUN).as_posix() for path in smoke.RUN.rglob('*')
               if path.is_file() and path.name.startswith('frame_registration_')]
    smoke.save(smoke.UNIT / 'registration_cleanup.json',
               {'passed': not scratch, 'unpublishedRegistrationScratchFiles': scratch,
                'scope': 'Owned registration database and all prefixed sidecars after successful pipeline exit.'})
    if scratch:
        raise ValueError('Successful pipeline retained unpublished registration scratch')
    return retained_compare_artifacts(helper)


smoke.compare_artifacts = compare_registration_artifacts

if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    raise SystemExit(smoke.main())
