"""Prepare exclusive revision 3; preserve reviewed/drafted revision 2 exactly."""
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
    prepared = json.loads((HERE / 'preparation_v2.json').read_text(encoding='utf-8'))
    history_names = ['document_evidence_v2.py', 'verify_installation_v2.py', 'seal_preparation_v2.py',
                     'preparation_v2.json', 'revision_2_origin.json', 'prepare_revision_2.py']
    history = {leaf: binding(HERE / leaf) for leaf in history_names}
    history.update({path.relative_to(HERE).as_posix(): binding(path)
                    for path in (HERE / 'draft_review_v2').rglob('*') if path.is_file()})
    for leaf in ('document_evidence_v2.py', 'verify_installation_v2.py', 'seal_preparation_v2.py'):
        assert history[leaf] == prepared['staticReview'][leaf]['binding']
    document = (HERE / 'document_evidence_v2.py').read_text(encoding='utf-8')
    document = replace_once(document, "PREPARATION = HERE / 'preparation_v2.json'", "PREPARATION = HERE / 'preparation_v3.json'")
    document = replace_once(document, "draft = HERE / 'draft_review_v2'", "draft = HERE / 'draft_review_v3'")
    old = ("'Rendering and final raw exporters read bounded snapshots; successful completion removes only owned scratch files, while failure closes and retains diagnostic rows. '")
    new = ("'Rendering and final raw exporters read bounded snapshots; successful completion removes only owned scratch files. '")
    new += ("\n            'Caught failures also close handles and remove owned temporary databases; existing logs and partial exports retain their prior failure behavior. '")
    document = replace_once(document, old, new)
    wrapper = (HERE / 'verify_installation_v2.py').read_text(encoding='utf-8').replace("HERE / 'preparation_v2.json'", "HERE / 'preparation_v3.json'")
    seal = (HERE / 'seal_preparation_v2.py').read_text(encoding='utf-8')
    for old_leaf, new_leaf in (('document_evidence_v2.py', 'document_evidence_v3.py'),
                              ('verify_installation_v2.py', 'verify_installation_v3.py'),
                              ('seal_preparation_v2.py', 'seal_preparation_v3.py'),
                              ('preparation_v2.json', 'preparation_v3.json')):
        seal = seal.replace(old_leaf, new_leaf)
    seal = replace_once(seal, "    value = {'preparedAt':", '''    for leaf in ('document_evidence_v2.py', 'verify_installation_v2.py', 'seal_preparation_v2.py',
                 'preparation_v2.json', 'revision_2_origin.json', 'prepare_revision_2.py',
                 'revision_3_origin.json', 'prepare_revision_3.py'):
        add(HERE / leaf)
    for path in (HERE / 'draft_review_v2').rglob('*'):
        if path.is_file():
            add(path)
    value = {'preparedAt':''')
    values = {'document_evidence_v3.py': document, 'verify_installation_v3.py': wrapper,
              'seal_preparation_v3.py': seal}
    assert not any((HERE / leaf).exists() for leaf in list(values) + ['revision_3_origin.json'])
    for leaf, text in values.items():
        ast.parse(text, filename=leaf)
        with (HERE / leaf).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    assert history == {leaf: binding(HERE / leaf) for leaf in history}
    with (HERE / 'revision_3_origin.json').open('x', encoding='utf-8') as stream:
        json.dump({'at': datetime.now(timezone.utc).isoformat(), 'status': 'PREPARED_NOT_EXECUTED',
            'revision2FilesIncludingDrafts': history, 'revision2FilesUnchanged': True,
            'newHelpers': {leaf: binding(HERE / leaf) for leaf in values},
            'reason': 'Correct reviewed lifecycle wording: caught IdentityFrameSequence failures remove exact owned scratch instead of retaining diagnostic rows. Prior v2 drafts were not published.',
            'guardsOrProductionChanged': False, 'documentationPublished': False,
            'hostVerifierRun': False, 'inferenceRun': False}, stream, indent=2)
    print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'revision2FilesUnchanged': True,
                      'helpers': {leaf: binding(HERE / leaf) for leaf in values}}))


if __name__ == '__main__':
    main()
