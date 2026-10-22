"""Integrity and leakage controls for bounded publisher archive acquisition."""
import io
import json
import zipfile

import pytest

from scripts.data.acquire_independent_tennis import (
    ARCHIVE_BYTES, ARCHIVE_MD5, ARCHIVE_NAME, RECORD_ID,
    RangeReader, check_record, check_reservation, safe_destination,
)


class Response:
    def __init__(self, content, header, status=206):
        self.status_code = status
        self.headers = {"Content-Range": header}
        self.raw = io.BytesIO(content)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, content, *, status=206, extra=b""):
        self.content, self.status, self.extra = content, status, extra

    def get(self, url, *, headers, stream, timeout):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return Response(self.content[start:end + 1] + self.extra,
                        f"bytes {start}-{end}/{len(self.content)}", self.status)


def test_zip_ranges_preserve_exact_original_member_crc_and_bytes():
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("MUVY/Tennis/Event_01/metadata.txt", "original source id")
    payload = source.getvalue()
    reader = RangeReader(Session(payload), size=len(payload), budget=4096)
    with zipfile.ZipFile(reader) as archive:
        assert archive.read("MUVY/Tennis/Event_01/metadata.txt") == b"original source id"
    assert sum(r["bytes"] for r in reader.ranges) == reader.received <= 4096
    assert all(len(r["sha256"]) == 64 for r in reader.ranges)


@pytest.mark.parametrize("kwargs", [{"status": 200}, {"extra": b"surplus"}])
def test_range_requires_exact_206_and_length(kwargs):
    reader = RangeReader(Session(b"012345", **kwargs), size=6, budget=10)
    with pytest.raises(ValueError):
        reader.read(4)
    assert reader.received == 0


def test_transfer_budget_counts_repeated_reads():
    reader = RangeReader(Session(b"012345"), size=6, budget=5)
    assert reader.read(4) == b"0123"
    reader.seek(0)
    with pytest.raises(ValueError, match="cumulative"):
        reader.read(2)


@pytest.mark.parametrize("name", ["../secret", "/outside", "Tennis/../../outside", "C:/secret", "Tennis\\secret"])
def test_untrusted_member_cannot_leave_workspace(tmp_path, name):
    with pytest.raises(ValueError):
        safe_destination(tmp_path, name)


def publisher_record():
    return {"id": RECORD_ID, "metadata": {"license": {"id": "cc-by-4.0"}},
            "files": [{"key": ARCHIVE_NAME, "size": ARCHIVE_BYTES, "checksum": "md5:" + ARCHIVE_MD5}]}


def test_publisher_data_license_and_revision_are_pinned():
    record = publisher_record()
    check_record(record)
    record["metadata"]["license"]["id"] = "MIT"
    with pytest.raises(ValueError, match="license"):
        check_record(record)
    record = publisher_record()
    record["files"][0]["size"] += 1
    with pytest.raises(ValueError, match="revision"):
        check_record(record)


def test_reservation_requires_original_match_not_random_clip_group(tmp_path):
    path = tmp_path / "reserved.json"
    document = {"reserved_before_visual_review": True, "qualification_evidence": False,
                "groups": [{"original_match_id": "2019-wimbledon-federer-nishikori-qf",
                            "source_video_ids": ["maRB027fzxc", "vnXVmFWk0nc"],
                            "split": "outside_scope_negative", "members": ["Tennis/Event_01/a.mp4"]}]}
    path.write_text(json.dumps(document))
    assert check_reservation(path, "Tennis/Event_01/a.mp4")["split"] == "outside_scope_negative"
    document["groups"][0].pop("original_match_id")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="original-match"):
        check_reservation(path, "Tennis/Event_01/a.mp4")


@pytest.mark.parametrize("change", [
    {"reserved_before_visual_review": False}, {"qualification_evidence": True},
])
def test_reservation_cannot_claim_inspected_source_is_frozen_unseen(tmp_path, change):
    path = tmp_path / "reserved.json"
    path.write_text(json.dumps({"reserved_before_visual_review": True,
                                "qualification_evidence": False, **change}))
    with pytest.raises(ValueError, match="pre-review"):
        check_reservation(path, "Tennis/Event_01/a.mp4")


def test_new_camera_from_development_match_cannot_be_frozen_test(tmp_path):
    path = tmp_path / "reserved.json"
    group = {"original_match_id": "2019-wimbledon-final", "source_video_ids": ["new-camera"],
             "split": "frozen_test_candidate", "members": ["Tennis/Event_02/new-camera.mp4"],
             "original_match_previously_used_for_development": True,
             "original_match_overlap_audit": "prior UVY final source was used for tuning"}
    document = {"reserved_before_visual_review": True, "qualification_evidence": False, "groups": [group]}
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="untouched original-match"):
        check_reservation(path, "Tennis/Event_02/new-camera.mp4")
    group["split"] = "outside_scope_negative"
    path.write_text(json.dumps(document))
    assert check_reservation(path, "Tennis/Event_02/new-camera.mp4")["original_match_previously_used_for_development"] is True
