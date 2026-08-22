"""
Unit tests for Phase 6.4 cross-match diagnostic integrity.
Validates media integrity, scientific split metadata, output schema versioning,
and nullable analytics handling.
"""

import os
import json
import hashlib
import pytest


def test_cross_match_media_integrity_and_hashes():
    """Verifies that all cross-match diagnostic video files have valid hashes."""
    meta_path = "data/benchmarks/cross_match_final_holdout/videos.json"
    assert os.path.exists(meta_path), "Cross-match videos.json missing"
    
    with open(meta_path, "r") as f:
        videos = json.load(f)["videos"]
        
    assert len(videos) == 3, f"Expected 3 diagnostic videos, found {len(videos)}"
    
    for vid_id, meta in videos.items():
        vpath = meta["path"]
        assert os.path.exists(vpath), f"Video file missing: {vpath}"
        assert meta["fps"] > 0, "Invalid FPS"
        assert meta["width"] > 0 and meta["height"] > 0, "Invalid resolution"
        assert meta["frame_count"] > 0, "Invalid frame count"
        
        # Verify SHA256
        h = hashlib.sha256()
        with open(vpath, "rb") as vf:
            while chunk := vf.read(8192):
                h.update(chunk)
        assert h.hexdigest() == meta["sha256"], f"SHA256 hash mismatch for {vid_id}"


def test_cross_match_split_is_not_mislabeled_as_pristine():
    """The inspected/tuned cross-match videos must never re-enter pristine GT."""
    splits_path = "data/benchmarks/cross_match_final_holdout/splits.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    diagnostic = set(splits["cross_match_diagnostic"])
    pristine = set(splits["pristine_final_holdout"])
    assert diagnostic == {"video_08", "video_09", "video_10"}
    assert not diagnostic.intersection(pristine)
    assert splits["qualification_evidence"] is False


def test_analytics_output_contract_schema_versioning():
    """Verifies that Analytics Output Contract V1 defines mandatory schema versioning."""
    contract_path = "docs/architecture/ANALYTICS_OUTPUT_CONTRACT_V1.md"
    assert os.path.exists(contract_path), "Contract V1 document missing"
    
    with open(contract_path, "r") as f:
        content = f.read()
        
    assert "schema_version" in content
    assert "1.0" in content
    assert "shot_events.json" in content
    assert "rallies.json" in content
    assert "line_calls.json" in content


def test_nullable_analytics_specification():
    """Verifies that contract forbids fabricated zeroes and mandates null for unobserved fields."""
    contract_path = "docs/architecture/ANALYTICS_OUTPUT_CONTRACT_V1.md"
    with open(contract_path, "r") as f:
        content = f.read()
        
    assert "No Fabricated Zeroes" in content
    assert "null" in content
