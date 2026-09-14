import pytest

from scripts.data.acquire_coco_tennis_validation import select_tennis_context


def sources():
    instances = {
        'categories': [{'id': 1, 'name': 'person'}, {'id': 43, 'name': 'tennis racket'}],
        'images': [{'id': i, 'width': 640, 'height': 480, 'file_name': f'{i}.jpg'} for i in (1, 2, 3)],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 43},
            {'id': 2, 'image_id': 1, 'category_id': 1, 'iscrowd': 1},
            {'id': 3, 'image_id': 2, 'category_id': 43},
            {'id': 4, 'image_id': 3, 'category_id': 1, 'iscrowd': 0}],
        'licenses': [{'id': 1, 'name': 'retained-source-license'}]}
    poses = {'categories': [{'id': 1, 'name': 'person'}],
             'images': list(instances['images']),
             'annotations': [{'id': 2, 'image_id': 1, 'num_keypoints': 0, 'iscrowd': 1}]}
    return instances, poses


def test_tennis_selection_keeps_crowds_and_images_without_eligible_poses():
    instances, poses = sources()
    ids, selected, selected_poses = select_tennis_context(instances, poses)
    assert ids == [1, 2]
    assert [row['id'] for row in selected['annotations']] == [1, 2, 3]
    assert selected['annotations'][1]['iscrowd'] == 1
    assert selected_poses['annotations'][0]['num_keypoints'] == 0
    assert [row['id'] for row in selected_poses['images']] == [1, 2]
    assert selected['licenses'] == instances['licenses']
    assert len(instances['images']) == 3


def test_selection_rejects_contiguous_or_incorrect_category_numbers():
    instances, poses = sources()
    instances['categories'][1]['id'] = 38
    with pytest.raises(ValueError, match='sparse COCO'):
        select_tennis_context(instances, poses)


def test_selection_rejects_inconsistent_image_geometry():
    instances, poses = sources()
    poses['images'] = [dict(row) for row in poses['images']]
    poses['images'][0]['width'] = 1280
    with pytest.raises(ValueError, match='metadata differ'):
        select_tennis_context(instances, poses)
