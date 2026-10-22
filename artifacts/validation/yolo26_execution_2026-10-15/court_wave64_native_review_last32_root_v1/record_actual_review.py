"""Record root's completed native-image and all-slot visual review, no inference."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
AIDS = ROOT / 'artifacts/validation/yolo26_execution_2026-10-15/court_wave64_native_review_aids_v1/review_aids_manifest.json'
EXPECTED = '3f4112bb190e6e8277a3406ea329d95868356237745c97f44ac01a9056729366'

# These observations were made by root viewing every original overlay and its
# unmarked, native 14x96-pixel detail board. They are not automatic decisions.
NOTES = {
33: ('Madrid', 'clay', [], 'MADRID and MUTUA MADRID OPEN visible; singles, all fourteen intersections visible and aligned.'),
34: ('Miami', 'hard', [], 'Miami/Itaú back wall and Isner/Federer scoreboard; elevated singles, all slots visible.'),
35: ('Shanghai', 'hard', [], 'SHANGHAI and Rolex Shanghai Masters visible; near player beside slot7, intersection remains discernible.'),
36: ('Miami', 'hard', [], 'Miami Open back wall and Khachanov/Anderson scoreboard; singles, sun/shadow boundary, all slots visible.'),
37: ('Dubai', 'hard', [], 'DUBAI and Dubai Duty Free branding; singles, all intersections visible.'),
38: ('Montreal', 'hard', [], 'MONTREAL375 and Rogers branding; singles, slot7 near moving player remains visible.'),
39: ('IndianWells', 'hard', [], 'INDIAN WELLS visible; lower quality image but fourteen white intersections discernible.'),
40: ('Washington', 'hard', [], 'WASHINGTON D.C.; Murray/Murray versus Mahut/Roger-Vasselin and four players: doubles, excluded from initial singles scope.'),
41: ('Miami', 'hard', [5], 'Itaú/betway Miami venue; elevated singles; near player covers or makes left singles-baseline intersection ambiguous.'),
42: ('Beijing', 'hard', [], 'BEIJING, China Open and Norrie/Murray visible; all fourteen intersections discernible.'),
43: ('IndianWells', 'hard', [], 'INDIAN WELLS and BNP PARIBAS visible; elevated slightly skewed singles view, all intersections visible.'),
44: ('Rome', 'clay', [], 'ROMA/BNL visible; singles, all intersections discernible including far service center above net.'),
45: ('Toronto', 'hard', [], 'TORONTO; Bryan/Sock versus Lopez/Lopez and four players: doubles, excluded.'),
46: ('Rotterdam', 'hard', [], 'ABN AMRO Rotterdam branding; extra player-like figure within far playing half plus far-baseline and near players. Frame composition/transition is unresolved despite singles scoreboard; excluded without assigning a replay explanation.'),
47: ('Dubai', 'hard', [], 'DUBAI/Dubai Duty Free branding; night singles, all fourteen intersections discernible.'),
48: ('MonteCarlo', 'clay', [], 'MONTE CARLO/FEDCOM visible; singles, all intersections visible; existing DEV constraint retained.'),
49: ('Paris', 'hard', [], 'PARIS visible; indoor singles, all fourteen intersections aligned.'),
50: ('Rotterdam', 'hard', [], 'ABN AMRO World Tennis Tournament circular branding; singles, all intersections visible.'),
51: ('Sydney', 'hard', [], 'SYDNEY/ATP CUP visible; Thiem/Hurkacz singles, all intersections visible.'),
52: ('Montreal', 'hard', [], 'MONTREAL375/Rogers visible; del Potro/Isner singles, all intersections discernible.'),
53: ('MonteCarlo', 'clay', [], 'MONTE CARLO/FEDCOM visible; Nishikori/Cilic singles, far player beside slot0 but corner remains visible; existing DEV constraint.'),
54: ('Miami', 'hard', [], 'Crandon Park/Miami back wall; Gasquet/Murray singles, all fourteen slots visible.'),
55: ('Cincinnati', 'hard', [], 'Western & Southern/UC Health visible; Murray/Gasquet singles, all intersections visible.'),
56: ('Montreal', 'hard', [], 'MONTREAL375/Rogers visible; singles, subtitle outside playing court does not cover a landmark.'),
57: ('Toronto', 'hard', [12], 'TORONTO/Rogers visible; Polansky/Ebden singles, far service-center intersection is covered by far player.'),
58: ('Rotterdam', 'hard', [], 'ABN AMRO circular World Tennis Tournament branding; De Minaur/Millman singles, all fourteen slots visible.'),
59: ('Miami', 'hard', [], 'Crandon Park/Miami back wall; Haase/del Potro singles, all fourteen intersections visible.'),
60: ('IndianWells', 'hard', [], 'INDIAN WELLS; Mektic/Pavic versus Isner/Sock and four players: doubles, excluded.'),
61: ('IndianWells', 'hard', [], 'INDIAN WELLS/BNP PARIBAS; del Potro/Djokovic singles, all fourteen slots discernible.'),
62: ('Toronto', 'hard', [5], 'TORONTO/Rogers; Djokovic/Stepanek singles, near left player shoe makes left singles-baseline slot ambiguous.'),
63: ('Auckland', 'hard', [], 'AUCKLAND/ASB visible; singles, all fourteen intersections discernible.'),
64: ('Miami', 'hard', [], 'Miami Open back wall; Tiafoe/Edmund singles, all fourteen intersections visible.'),
}
REJECT = {40: 'DOUBLES_OUTSIDE_INITIAL_SCOPE', 45: 'DOUBLES_OUTSIDE_INITIAL_SCOPE',
          46: 'UNRESOLVED_EXTRA_PLAYER_FRAME_COMPOSITION', 60: 'DOUBLES_OUTSIDE_INITIAL_SCOPE'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if digest(AIDS) != EXPECTED:
        raise ValueError('Unexpected review-aid manifest bytes')
    aids = json.loads(AIDS.read_text(encoding='utf-8-sig'))
    selected = aids['samples'][32:64]
    assert len(selected) == 32 and set(NOTES) == {s['number'] for s in selected}
    verified = []
    rows = []
    for source in selected:
        number = source['number']
        context, surface, withheld, note = NOTES[number]
        for key in ('source_image', 'native_review_overlay', 'point_detail_board'):
            ref = source[key]
            if digest(ROOT / ref['path']) != ref['sha256']:
                raise ValueError(f'Changed {key}: {source["id"]}')
            verified.append(ref)
        assert not source['out_of_image_keypoint_indices']
        coords = source['publisher_keypoints_xy_px']
        assert len(coords) == 14 and all(len(p) == 2 for p in coords)
        accepted = number not in REJECT
        slots = []
        for slot, point in enumerate(coords):
            slots.append({'slot': slot, 'semantic_name': aids['point_names'][slot],
                          'publisher_xy_px': point,
                          'visibility_state': 'NULL_OCCLUDED_OR_AMBIGUOUS' if slot in withheld else 'VISIBLE',
                          'accepted_xy_px': point if accepted and slot not in withheld else None,
                          'actually_reviewed_unmarked_native_patch': True})
        rows.append({'number': number, 'id': source['id'], 'source_video_id': source['source_video_id'],
                     'source_frame_ordinal': source['source_frame_ordinal'],
                     'source_pts': None, 'source_time_base': None, 'source_fps': None,
                     'match_id': None, 'camera_session_id': None,
                     'reviewer': 'ROOT_AGENT', 'independent_human_review': False,
                     'decision': 'ACCEPT_PENDING_CONTEXT_UNION' if accepted else 'REJECT',
                     'reason': REJECT.get(number), 'visual_event_family': context,
                     'surface': surface, 'development_split': None, 'context_group': None,
                     'actual_native_overlay_viewed': True, 'actual_all14_unmarked_native_patches_viewed': True,
                     'source_image': source['source_image'], 'native_review_overlay': source['native_review_overlay'],
                     'point_detail_board': source['point_detail_board'], 'point_review': slots,
                     'reviewed_keypoints_px': [p['accepted_xy_px'] for p in slots] if accepted else None,
                     'observations': note, 'publisher_coordinates_modified': False,
                     'localization_uncertainty_px': None,
                     'localization_uncertainty_note': 'Visual alignment accepted for development; no independently calibrated pixel uncertainty.'})
    visible = sum(p['accepted_xy_px'] is not None for r in rows for p in r['point_review'])
    accepted = [r for r in rows if r['decision'] == 'ACCEPT_PENDING_CONTEXT_UNION']
    review = {'schema_version': '1.0', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'ACTUAL_LAST32_ROOT_NATIVE_AND14_SLOT_REVIEW_COMPLETE_PENDING_CONTEXT_UNION',
              'review_aids_manifest': {'path': str(AIDS), 'sha256': EXPECTED},
              'recording_script_sha256': digest(Path(__file__)), 'reviewer': 'ROOT_AGENT',
              'counts': {'images_actually_viewed': 32, 'native_overlays_viewed': 32,
                         'unmarked14point_boards_viewed': 32, 'slots_actually_reviewed': 448,
                         'accepted': len(accepted), 'rejected': 32-len(accepted),
                         'accepted_visible_coordinates': visible, 'accepted_null_slots': len(accepted)*14-visible},
              'samples': rows, 'source_and_aid_hashes_verified': verified,
              'final_splits_assigned': False, 'qualification_evidence': False,
              'independent_human_review': False, 'training_performed': False,
              'labels_exported': False, 'final_test_used': False,
              'split_rule': 'Retain all old165 event/source/compilation/rejected links; union before any split. Any prior DEV component stays DEV. No new TEST.'}
    with (BASE/'review.json').open('x', encoding='utf-8') as out:
        json.dump(review, out, indent=2, allow_nan=False)
    print(json.dumps({'counts': review['counts'], 'review_sha256': digest(BASE/'review.json')}))

if __name__ == '__main__':
    main()
