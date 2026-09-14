import json

import numpy as np
import pytest

from scripts.evaluate.replay_ball_temporal_detours import digest
from scripts.train.train_feature_ball_verifier import load_features
from src.detection.feature_ball_verifier import ENCODER_SHA256


def make_cache(tmp_path):
    original = tmp_path/'original'
    original.mkdir()
    np.savez(original/'candidates.npz',features=np.array([[.6,1.]],np.float32),targets=np.array([0],np.int64))
    source = {'split':'TRAINING_ONLY','candidate_count':1,'cache_sha256':digest(original/'candidates.npz'),
              'rows':[{'clip':'a_000','frame':9,'candidates':[{'sample_index':0,'training_target':0}]}]}
    (original/'manifest.json').write_text(json.dumps(source))
    features = tmp_path/'features'
    features.mkdir()
    np.save(features/'appearance.npy',np.zeros((1,6144),np.float16),allow_pickle=False)
    manifest = {'complete':True,'encoder_sha256':ENCODER_SHA256,'code_hashes':{},'source_manifest':str(original/'manifest.json'),
                'source_manifest_sha256':digest(original/'manifest.json'),'feature_sha256':digest(features/'appearance.npy'),
                'split':'TRAINING_ONLY','candidate_count':1,'completed_clips':['a_000']}
    (features/'manifest.json').write_text(json.dumps(manifest))
    return features, manifest, source, original


def test_verified_features_keep_original_metadata_and_labels(tmp_path):
    directory, _, _, _ = make_cache(tmp_path)
    _, _, appearance, metadata, targets = load_features(directory)
    assert appearance.shape==(1,6144) and appearance.dtype==np.float16
    assert metadata[0,1]==1 and targets.tolist()==[0]


@pytest.mark.parametrize('change',['incomplete','duplicate_clip','wrong_encoder','wrong_feature_hash'])
def test_incomplete_or_changed_cache_rejected(tmp_path,change):
    directory, manifest, _, _ = make_cache(tmp_path)
    if change=='incomplete':
        manifest['complete']=False
    elif change=='duplicate_clip':
        manifest['completed_clips']*=2
    elif change=='wrong_encoder':
        manifest['encoder_sha256']='wrong'
    else:
        manifest['feature_sha256']='wrong'
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_features(directory)


def test_source_label_mapping_must_match_cache(tmp_path):
    directory, manifest, source, original = make_cache(tmp_path)
    source['rows'][0]['candidates'][0]['training_target']=1
    (original/'manifest.json').write_text(json.dumps(source))
    manifest['source_manifest_sha256']=digest(original/'manifest.json')
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='target mapping'):
        load_features(directory)


def test_duplicate_labeled_rows_rejected(tmp_path):
    directory, manifest, source, original = make_cache(tmp_path)
    source['rows'] *= 2
    (original/'manifest.json').write_text(json.dumps(source))
    manifest['source_manifest_sha256'] = digest(original/'manifest.json')
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Duplicate labeled'):
        load_features(directory)
