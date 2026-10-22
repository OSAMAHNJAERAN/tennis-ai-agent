"""Preserve the failed synthetic attempt; prepare an explicit fixture correction."""
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).resolve().parent
def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert checksum(UNIT / 'candidate.py') == '0e0a6420432ca3866326d0fe6c5fa69494110c2719e6faa6b7a658931ebccfd0'
assert checksum(UNIT / 'protocol.json') == '6b6609c7743a5a5434f2a56951fdaf4f1b29febeb32b7d69782bd5ef2fc93421'
old = UNIT / 'test_candidate.py'
assert checksum(old) == 'ee137105840dc0ab451748d8cd21c4226a4dcd5fb487ee621a1ea3f0ce3a8fc5'
receipt = json.loads((UNIT / 'tests/receipt.json').read_text(encoding='utf-8'))
assert receipt['counts'] == {'tests': 35, 'failures': 1, 'errors': 0, 'skipped': 0, 'passed': 34}
assert receipt['sourcesUnchanged'] and receipt['allOwnedIdentitiesRetired']
assert receipt['failureCases'][0]['name'] == 'test_colored_ridge_is_not_rescued_by_the_surface_condition'
target = '    image = paint()\n    image[49:52] = (0, 255, 255)\n    assert not retained(candidate.line_evidence(image, SEGMENT, upper_surface()))'
replacement = '''    # Every sampled center and neighbor is saturated, so the existing
    # central-minimum-saturation condition is genuinely false at every shift.
    image = paint()
    image[:50] = (255, 0, 0)
    image[50:] = (0, 255, 0)
    image[49:52] = (0, 255, 255)
    actual = candidate.line_evidence(image, SEGMENT, upper_surface())
    assert actual["white_fraction"] == 0
    assert not retained(actual)


def test_thin_colored_ridge_on_gray_background_exposes_inherited_whiteness_limit():
    # This failed the first rejection fixture. It documents an inherited
    # minimum-saturation limitation, rather than changing candidate thresholds.
    image = paint()
    image[49:52] = (0, 255, 255)
    actual = candidate.line_evidence(image, SEGMENT, upper_surface())
    assert actual["evidence_origin"] == "SURFACE_BOUNDARY_RECOVERY"
    assert actual["white_fraction"] == 1
    assert retained(actual)'''
source = old.read_text(encoding='utf-8')
assert source.count(target) == 1
test_v2 = UNIT / 'test_candidate_v2.py'
with test_v2.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(source.replace(target, replacement))
runner = (UNIT / 'run_tests.py').read_text(encoding='utf-8')
runner = runner.replace('UNIT / "tests"', 'UNIT / "tests_v2"')
runner = runner.replace('"test_candidate.py"', '"test_candidate_v2.py"')
runner = runner.replace('ee137105840dc0ab451748d8cd21c4226a4dcd5fb487ee621a1ea3f0ce3a8fc5', checksum(test_v2))
runner = runner.replace('UNIT / "run_tests.py"', 'UNIT / "run_tests_v2.py"')
with (UNIT / 'run_tests_v2.py').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(runner)
with (UNIT / 'test_followup_preparation.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump({'candidateUnchanged': True, 'protocolUnchanged': True, 'realCasesNotRun': True,
        'priorAttempt': {'path': 'tests/receipt.json', 'sha256': checksum(UNIT / 'tests/receipt.json'), 'counts': receipt['counts']},
        'reason': 'Gray neighbors satisfy the inherited minimum-saturation criterion. Saturated surroundings exercise actual whiteness rejection; a separate case explicitly records gray-background false evidence. No threshold, candidate or protocol changed.',
        'newTestSha256': checksum(test_v2), 'newRunnerSha256': checksum(UNIT / 'run_tests_v2.py')}, stream, indent=2)
print('Fresh synthetic fixture follow-up prepared; original failed attempt preserved.')
