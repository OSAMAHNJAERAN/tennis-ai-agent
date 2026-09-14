"""Validate training-only uncertainty exclusions without changing cached labels."""


def reviewed_negative_indices(manifest, review):
    if manifest['split'] != 'TRAINING_ONLY' or not review.get('complete'):
        raise ValueError('Completed training-only review required')
    if review.get('scope') != 'TRAINING_ONLY_NEGATIVE_UNCERTAINTY_QUARANTINE_NO_RELABELING' or review.get('labels_changed') is not False:
        raise ValueError('Review must quarantine uncertainty without relabeling')
    clips = ['_'.join(item) for item in manifest['selected_clips']]
    if len(clips) != 20 or len(set(clips)) != 20:
        raise ValueError('Expected original twenty-clip partition')
    train = set(clips[:16])
    candidates = {}
    for row in manifest['rows']:
        for candidate in row['candidates']:
            index = candidate['sample_index']
            if index in candidates:
                raise ValueError('Duplicate cache sample index')
            candidates[index] = row, candidate
    seen, excluded = set(), set()
    for item in review['reviews']:
        index = item['candidate']['sample_index']
        if index in seen or index not in candidates:
            raise ValueError('Duplicate or unknown review sample')
        seen.add(index)
        row, candidate = candidates[index]
        if row['clip'] not in train or row['target_xy'] is not None or candidate['training_target'] != 0:
            raise ValueError('Only actual training absence negatives may be reviewed')
        if (item['clip'], item['frame'], item['candidate']) != (row['clip'], row['frame'], candidate):
            raise ValueError('Review candidate identity changed')
        judgment = item['judgment']
        if judgment not in ('apparent_moving_ball', 'unresolved', 'clear_non_ball_distractor'):
            raise ValueError('Unknown review judgment')
        expected = judgment in ('apparent_moving_ball', 'unresolved')
        if item['exclude_from_training_loss'] is not expected:
            raise ValueError('Exclusion violates frozen uncertainty policy')
        if expected:
            excluded.add(index)
    return excluded
