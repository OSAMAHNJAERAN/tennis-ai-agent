"""Actual audit writer source blocks close independently of inference owners."""
import ast
from contextlib import ExitStack
import json
from pathlib import Path

import pytest

from src.utils.json_array_artifact import JsonArrayArtifact


PATH = Path(__file__).resolve().parents[1] / 'src/pipeline/phase6_pipeline.py'
NAMES = ('camera_registration.json', 'camera_intervals.json', 'match_identity.json', 'image_people.json')


def parts():
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
               and node.name == '_run_with_frame_exports')
    players = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
                   and node.name == '_candidate_player_pass')
    prefix_branch = next(node for node in run.body if isinstance(node, ast.If)
                         and ast.unparse(node.test) == 'prefix is not None')
    fresh = prefix_branch.orelse

    def assignment(body, name):
        return next(node for node in body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))

    setup = assignment(fresh, 'player_audits')
    writers = [assignment(fresh, 'camera_audit'), assignment(players.body, 'interval_audit'),
               assignment(players.body, 'identity_audit'), assignment(fresh, 'people_audit')]
    close = next(node for node in fresh if isinstance(node, ast.If)
                 and 'player_audits.close()' in ast.unparse(node))
    return run, players, setup, writers, close


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), namespace)


def namespace(tmp_path, stack, *, disk=True):
    owners = []

    class ObservedWriter(JsonArrayArtifact):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exits = 0
            owners.append(self)

        def __exit__(self, *args):
            self.exits += 1
            return super().__exit__(*args)

    import os
    values = dict(export_stack=stack, disk_players=disk, ExitStack=ExitStack, os=os,
                  output_dir=str(tmp_path), recovery_enabled=False, camera_enabled=True,
                  camera_settings={}, recovery_settings={}, analysis_id='synthetic-audit-only',
                  JsonArrayArtifact=ObservedWriter, image_people_metadata=lambda: {'schemaVersion': 1})
    return values, owners


def setup_writers(values):
    _, _, setup, writers, _ = parts()
    execute([setup], values)
    values['audit_stack'] = values['player_audits']
    execute(writers, values)
    for name in ('camera_audit', 'interval_audit', 'identity_audit', 'people_audit'):
        values[name].append({'frame_index': 0, 'scope': 'SYNTHETIC_ONLY'})


def test_player_audits_publish_before_main_owners_close_without_second_exit(tmp_path):
    indexed_owner_closures = []
    with ExitStack() as stack:
        stack.callback(lambda: indexed_owner_closures.append('main storage closed'))
        values, owners = namespace(tmp_path, stack)
        setup_writers(values)
        assert all(not (tmp_path / name).exists() for name in NAMES)
        execute([parts()[-1]], values)
        assert not indexed_owner_closures
        assert [writer.exits for writer in owners] == [1] * 4
        for name in NAMES:
            assert json.loads((tmp_path / name).read_text())['frames'] == [
                {'frame_index': 0, 'scope': 'SYNTHETIC_ONLY'}]
    assert indexed_owner_closures == ['main storage closed']
    assert [writer.exits for writer in owners] == [1] * 4


def test_player_failure_discards_all_private_audit_temporaries(tmp_path):
    failure = KeyboardInterrupt('injected player interruption')
    with pytest.raises(KeyboardInterrupt) as caught:
        with ExitStack() as stack:
            values, owners = namespace(tmp_path, stack)
            setup_writers(values)
            raise failure
    assert caught.value is failure
    assert [writer.exits for writer in owners] == [1] * 4
    assert list(tmp_path.iterdir()) == []


def test_historical_audits_keep_main_stack_lifetime(tmp_path):
    with ExitStack() as stack:
        values, owners = namespace(tmp_path, stack, disk=False)
        setup_writers(values)
        execute([parts()[-1]], values)
        assert values['player_audits'] is stack
        assert [writer.exits for writer in owners] == [0] * 4
        assert all(not (tmp_path / name).exists() for name in NAMES)
    assert [writer.exits for writer in owners] == [1] * 4


def test_actual_pipeline_keeps_observation_owners_on_main_stack():
    run, players, setup, writers, close = parts()
    prefix_branch = next(node for node in run.body if isinstance(node, ast.If)
                         and ast.unparse(node.test) == 'prefix is not None')
    assert 'export_stack.enter_context(ExitStack())' in ast.unparse(setup)
    assert all('player_audits.enter_context' in ast.unparse(node) for node in (writers[0], writers[3]))
    assert all('audit_stack.enter_context' in ast.unparse(node) for node in (writers[1], writers[2]))
    source = ast.unparse(players)
    assert 'export_stack.enter_context(IdentityFrameSequence(output_dir))' in source
    assert 'audit_stack = export_stack if audit_stack is None else audit_stack' in source
    invocation = next(node for node in prefix_branch.orelse if isinstance(node, ast.If)
                      and 'self._candidate_player_pass(' in ast.unparse(node))
    assert 'audit_stack=player_audits' in ast.unparse(invocation)
    transition = next(node for node in prefix_branch.orelse if isinstance(node, ast.If)
                      and 'finish_tracking_for_crops' in ast.unparse(node))
    ball_storage = next(node for node in run.body if isinstance(node, ast.Assign)
                        and 'BallCandidateSequence(' in ast.unparse(node))
    assert transition.lineno < close.lineno < ball_storage.lineno
