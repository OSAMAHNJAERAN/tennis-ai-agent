"""Prepare the pose/racket pipeline integration; --apply requires a genuine RED."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
PIPELINE = ROOT / 'src/pipeline/phase6_pipeline.py'
EXPECTED = '93b1bcf44a56ed70d76264ae7ab8cd2e8423616821632da09904d71ddc3c75ec'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepared_source():
    assert sha(PIPELINE) == EXPECTED
    original = PIPELINE.read_bytes().decode('utf-8')
    text = original
    newline = '\r\n' if '\r\n' in text else '\n'
    def replace(before, after):
        nonlocal text
        before, after = before.replace('\n', newline), after.replace('\n', newline)
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    replace('from contextlib import ExitStack', 'from contextlib import ExitStack, contextmanager')
    replace('from src.utils.frame_registration_sequence import FrameRegistration, FrameRegistrationSequence',
            'from src.utils.frame_registration_sequence import FrameRegistration, FrameRegistrationSequence\n'
            'from src.utils.identity_frame_sequence import IdentityFrameSequence, write_identity_frame_json')
    replace('def detection_export_records(p1_boxes, p2_boxes, ball_points, frame_calibrations):',
            '@contextmanager\n'
            'def releasing_video_handle(handle):\n'
            '    """Always release an owned render handle without replacing a primary error."""\n'
            '    try:\n'
            '        yield handle\n'
            '    except BaseException as error:\n'
            '        try:\n'
            '            handle.release()\n'
            '        except BaseException as cleanup_error:\n'
            '            error.add_note(f"Render handle release also failed: {cleanup_error}")\n'
            '        raise\n'
            '    else:\n'
            '        handle.release()\n\n\n'
            'def detection_export_records(p1_boxes, p2_boxes, ball_points, frame_calibrations):')
    replace('        pose_frames = []\n        racket_frames = []',
            '        pose_frames = export_stack.enter_context(IdentityFrameSequence(output_dir)) if self.motion_tracker is not None else []\n'
            '        racket_frames = export_stack.enter_context(IdentityFrameSequence(output_dir)) if self.racket_tracker is not None else []')
    replace('            "player_2": p2_motion,\n            "pose_frames": pose_frames,', '            "player_2": p2_motion,')
    start = text.index("        fourcc = cv2.VideoWriter_fourcc(*'mp4v')")
    end = text.index('        writer.release()' + newline, start) + len('        writer.release()' + newline)
    render = text[start:end]
    assert render.count('        cap.release()' + newline) == 1 and render.count('        writer.release()' + newline) == 1
    render = render.replace('        cap.release()' + newline, '').replace('        writer.release()' + newline, '')
    render = render.replace('        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))',
                            '        writer = render_stack.enter_context(releasing_video_handle(cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))))')
    render = render.replace('        cap = cv2.VideoCapture(input_video_path)',
                            '        cap = render_stack.enter_context(releasing_video_handle(cv2.VideoCapture(input_video_path)))')
    text = text[:start] + '        with ExitStack() as render_stack:' + newline + ''.join('    ' + line if line.strip() else line for line in render.splitlines(keepends=True)) + text[end:]
    replace('        with open(os.path.join(output_dir, "player_motion.json"), "w", encoding="utf-8") as f:\n'
            '            json.dump(_sanitize(player_motion), f, indent=2, allow_nan=False)\n'
            '        with open(os.path.join(output_dir, "racket_tracking.json"), "w", encoding="utf-8") as f:\n'
            '            json.dump(_sanitize({"schema_version": "1.0", "enabled": self.racket_tracker is not None,\n'
            '                                 "coordinate_system": "SOURCE_IMAGE_PIXELS", "accuracy_validated": False,\n'
            '                                 "motion_semantics": "BBOX_CENTER_TRANSLATION_NOT_SWING_OR_CONTACT_CLASSIFICATION",\n'
            '                                 "frames": racket_frames}), f, indent=2, allow_nan=False)',
            '        write_identity_frame_json(os.path.join(output_dir, "player_motion.json"), player_motion,\n'
            '                                  "pose_frames", pose_frames, _sanitize)\n'
            '        write_identity_frame_json(os.path.join(output_dir, "racket_tracking.json"),\n'
            '                                  {"schema_version": "1.0", "enabled": self.racket_tracker is not None,\n'
            '                                   "coordinate_system": "SOURCE_IMAGE_PIXELS", "accuracy_validated": False,\n'
            '                                   "motion_semantics": "BBOX_CENTER_TRANSLATION_NOT_SWING_OR_CONTACT_CLASSIFICATION"},\n'
            '                                  "frames", racket_frames, _sanitize)')
    ast.parse(text)
    return original.encode('utf-8'), text.encode('utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--red-junit')
    args = parser.parse_args()
    before, after = prepared_source()
    if not args.apply:
        with (BASE / 'pipeline_prepared.py').open('xb') as stream:
            stream.write(after)
        print(json.dumps({'parsed': True, 'productionChanged': False, 'preparedSha256': hashlib.sha256(after).hexdigest()}))
        return
    assert args.red_junit and not (BASE / 'integration_receipt.json').exists()
    report = (ROOT / args.red_junit).resolve()
    assert report.is_relative_to(BASE)
    suite = ET.parse(report).getroot()
    cases = list(suite.iter('testcase'))
    names = {case.get('name'): case for case in cases}
    retained = [case for name, case in names.items() if name.startswith('test_pose_and_racket_original_records_are_bounded_before_shot_linking')]
    assert len(retained) == 2 and all(case.find('failure') is not None for case in retained)
    assert any(case.find('failure') is not None and case.get('name').startswith('test_render_handles_release_on_failure') for case in cases)
    assert PIPELINE.read_bytes() == before
    PIPELINE.write_bytes(after)
    assert PIPELINE.read_bytes() == after
    with (BASE / 'integration_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump({'appliedAt': datetime.now(timezone.utc).isoformat(), 'beforeSha256': EXPECTED,
                   'afterSha256': sha(PIPELINE), 'helperSha256': sha(Path(__file__)),
                   'redJunit': report.relative_to(ROOT).as_posix(), 'redJunitSha256': sha(report),
                   'scope': 'Temporary pose/racket snapshots and explicit rendering handle cleanup; no threshold/model/authority change.'}, stream, indent=2)
    print(json.dumps({'productionChanged': True, 'sourceSha256': sha(PIPELINE)}))

if __name__ == '__main__':
    main()
