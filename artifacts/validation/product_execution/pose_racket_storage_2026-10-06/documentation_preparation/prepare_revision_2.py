"""Exclusively prepare revision 2; retain initial prepared helpers byte exact."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def replace_once(text, old, new):
    assert text.count(old) == 1, 'Revision anchor absent/nonunique'
    return text.replace(old, new, 1)


def main():
    prepared = json.loads((HERE / 'preparation.json').read_text(encoding='utf-8'))
    history = {leaf: binding(HERE / leaf) for leaf in (
        'document_evidence.py', 'verify_installation.py', 'seal_preparation.py',
        'preparation.json', 'initial_dry_run.json')}
    for leaf in ('document_evidence.py', 'verify_installation.py', 'seal_preparation.py'):
        assert history[leaf] == prepared['staticReview'][leaf]['binding']
    document = (HERE / 'document_evidence.py').read_text(encoding='utf-8')
    document = replace_once(document, "PREPARATION = HERE / 'preparation.json'", "PREPARATION = HERE / 'preparation_v2.json'")
    audit = '''def independent_measurement_gate():
    audit = load(UNIT / 'measurement/terminal_audit.json')
    require(audit['passed'] and audit['status'] == 'PASSED' and
            audit['allAuditBindingsUnchanged'] and audit['fullFileComparisonsIndependentlyPassed'] == 8 and
            audit['nativeRowsAuditedAcrossBothModes'] == 37200 and len(audit['runs']) == 4 and
            len(audit['comparisons']) == 2, 'Independent complete-row/byte/native/type/process audit')
    verify(audit['inputAndEvidenceBindings'])
    return {'passed': True, 'status': 'PASSED', 'fullFileComparisons': 8,
            'nativeRowsAcrossBothModes': 37200, 'receipt': name(UNIT / 'measurement/terminal_audit.json'),
            'receiptBinding': binding(UNIT / 'measurement/terminal_audit.json')}


'''
    document = replace_once(document, 'def worker_gate():\n', audit + 'def worker_gate():\n')
    document = replace_once(document, "choices.add_argument('--apply', action='store_true')", "choices.add_argument('--apply', action='store_true')\n    choices.add_argument('--draft-only', action='store_true')")
    document = replace_once(document, "'worker': worker_gate(), 'runtime': runtime_gate(),", "'worker': worker_gate(), 'runtime': runtime_gate(),\n                'independentMeasurementAudit': independent_measurement_gate(),")
    document = replace_once(document, "    if args.validate_only:\n", '''    if args.draft_only:
        draft = HERE / 'draft_review_v2'
        draft.mkdir(exist_ok=False)
        for pathname, data in outputs.items():
            with (draft / Path(pathname).name).open('xb') as stream:
                stream.write(data)
        save(draft / 'receipt.json', {'preparedAt': stamp(), 'status': 'DRAFT_ONLY',
             'evidence': evidence, 'inputBindings': TRACKED,
             'draftBindings': {Path(key).name: binding(draft / Path(key).name) for key in DOCS},
             'originalDocumentBindings': {key: binding(ROOT / key) for key in DOCS},
             'documentationPublished': False, 'hostVerifierRun': False,
             'modelsImported': False, 'inferenceRun': False, 'qualificationEvidence': False})
        verify(TRACKED, tracked=False)
        print(json.dumps({'passed': True, 'status': 'DRAFT_ONLY', 'draft': name(draft),
                          'documentationPublished': False, 'hostVerifierRun': False}))
        return 0
    if args.validate_only:
''')
    wrapper = (HERE / 'verify_installation.py').read_text(encoding='utf-8').replace("HERE / 'preparation.json'", "HERE / 'preparation_v2.json'")
    seal = (HERE / 'seal_preparation.py').read_text(encoding='utf-8')
    for old, new in (('document_evidence.py', 'document_evidence_v2.py'),
                     ('verify_installation.py', 'verify_installation_v2.py'),
                     ('seal_preparation.py', 'seal_preparation_v2.py'),
                     ('preparation.json', 'preparation_v2.json')):
        seal = seal.replace(old, new)
    # The top-level measurement preparation remains its actual historical name.
    # This replacement changes only local documentary helper paths; no other
    # evidence file names in the generic inventory are rewritten.
    seal = replace_once(seal, "    for leaf in static:\n        add(HERE / leaf)\n", '''    for leaf in static:
        add(HERE / leaf)
    for leaf in ('document_evidence.py', 'verify_installation.py', 'seal_preparation.py',
                 'preparation.json', 'initial_dry_run.json', 'revision_2_origin.json', 'prepare_revision_2.py'):
        add(HERE / leaf)
''')
    values = {'document_evidence_v2.py': document, 'verify_installation_v2.py': wrapper,
              'seal_preparation_v2.py': seal}
    for leaf in values:
        assert not (HERE / leaf).exists()
    assert not (HERE / 'revision_2_origin.json').exists()
    for leaf, text in values.items():
        ast.parse(text, filename=leaf)
        with (HERE / leaf).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    assert history == {leaf: binding(HERE / leaf) for leaf in history}
    with (HERE / 'revision_2_origin.json').open('x', encoding='utf-8') as stream:
        json.dump({'at': datetime.now(timezone.utc).isoformat(), 'status': 'PREPARED_NOT_EXECUTED',
            'initialPreparedFiles': history, 'initialPreparedFilesUnchanged': True,
            'newHelpers': {leaf: binding(HERE / leaf) for leaf in values},
            'reason': 'Add independent complete measurement terminal audit gate and review-only draft output. Preserve initial dry run/preparation/helpers unchanged.',
            'documentationPublished': False, 'hostVerifierRun': False, 'inferenceRun': False}, stream, indent=2)
    print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'initialFilesUnchanged': True,
                      'helpers': {leaf: binding(HERE / leaf) for leaf in values}}))


if __name__ == '__main__':
    main()
