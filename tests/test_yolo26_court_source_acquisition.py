"""Bounded range, PNG structure and pre-network refusal invariants."""

import json
import struct
import zlib

import pytest

from scripts.data import acquire_yolo26_court_source_wave as acquisition


def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def png(width=1280, height=720):
    # Structural fixture only: the implementation deliberately does not decode pixels.
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b"")


def member(*, flags=0, zip64=False, descriptor64=False, data=None):
    data = png() if data is None else data
    compressor = zlib.compressobj(wbits=-15)
    packed = compressor.compress(data) + compressor.flush()
    name = b"data/images/newvideo001_0.png"
    crc = zlib.crc32(data) & 0xFFFFFFFF
    extra = struct.pack("<HHQQ", 1, 16, len(data), len(packed)) if zip64 else b""
    compressed, uncompressed = (0xFFFFFFFF, 0xFFFFFFFF) if zip64 else (len(packed), len(data))
    header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 45 if zip64 else 20, flags, 8, 0, 0,
                         0 if flags & 8 else crc, compressed, uncompressed, len(name), len(extra))
    descriptor = b"PK\x07\x08" + struct.pack("<IQQ" if descriptor64 else "<III", crc, len(packed), len(data)) if flags & 8 else b""
    block = header + name + extra + packed + descriptor + b"\x00" * (1024 - len(extra) - len(descriptor))
    entry = {"name": name.decode(), "bytes": len(data), "compressed_bytes": len(packed),
             "crc32": crc, "planned_range_bytes_upper_bound": 30 + len(name) + 1024 + len(packed)}
    return block, entry


@pytest.mark.parametrize("flags,zip64,descriptor64", [(0, False, False), (2048, False, False),
                                                    (8, False, False), (8, False, True),
                                                    (0, True, False), (8, True, True)])
def test_native_png_member_header_crc_sizes_and_geometry(flags, zip64, descriptor64):
    block, entry = member(flags=flags, zip64=zip64, descriptor64=descriptor64)
    data, receipt = acquisition.verified_png_member(block, entry)
    assert data == png()
    assert receipt["native_png_geometry"] == [1280, 720]
    assert receipt["pixel_decoding_performed"] is False
    assert receipt["zip_local_header_sizes_crc_verified"] is True
    assert receipt["png_all_chunk_crcs_verified"] is True


@pytest.mark.parametrize("offset,value", [(0, 0), (6, 1), (6, 16), (8, 0), (30, 0)])
def test_wrong_header_encryption_method_and_member_name_fail(offset, value):
    block, entry = member()
    damaged = bytearray(block)
    damaged[offset] = value
    with pytest.raises(ValueError):
        acquisition.verified_png_member(bytes(damaged), entry)


def test_exact_range_length_is_required():
    block, entry = member()
    with pytest.raises(ValueError, match="range length"):
        acquisition.verified_png_member(block[:-1], entry)


def test_header_crc_must_match_pinned_index():
    block, entry = member()
    damaged = bytearray(block)
    damaged[14] ^= 1
    with pytest.raises(ValueError, match="sizes/CRC"):
        acquisition.verified_png_member(bytes(damaged), entry)


def test_data_descriptor_must_match_pinned_index():
    block, entry = member(flags=8)
    damaged = bytearray(block)
    payload_end = 30 + len(entry["name"]) + entry["compressed_bytes"]
    damaged[payload_end + 4] ^= 1
    with pytest.raises(ValueError, match="descriptor"):
        acquisition.verified_png_member(bytes(damaged), entry)


def test_deflate_payload_corruption_is_rejected():
    block, entry = member()
    damaged = bytearray(block)
    damaged[30 + len(entry["name"]) + 2] ^= 0x80
    with pytest.raises((ValueError, zlib.error)):
        acquisition.verified_png_member(bytes(damaged), entry)


@pytest.mark.parametrize("width,height", [(720, 1280), (1920, 1080), (1279, 720), (1280, 0)])
def test_resized_or_wrong_native_geometry_is_rejected(width, height):
    with pytest.raises(ValueError, match="1280x720"):
        acquisition.png_geometry(png(width, height))


def test_png_chunk_crc_is_checked_separately_from_zip_crc():
    damaged = bytearray(png())
    damaged[29] ^= 1
    block, entry = member(data=bytes(damaged))
    with pytest.raises(ValueError, match="chunk CRC"):
        acquisition.verified_png_member(block, entry)


@pytest.mark.parametrize("data", [b"not_png", png()[:-1], png() + b"trailing"])
def test_truncated_or_extra_png_structure_is_rejected(data):
    with pytest.raises(ValueError):
        acquisition.png_geometry(data)


@pytest.mark.parametrize("ram,disk", [(acquisition.RAM_FLOOR - 1, acquisition.DISK_FLOOR),
                                    (acquisition.RAM_FLOOR, acquisition.DISK_FLOOR - 1)])
def test_media_floors_remain_one_gib_ram_and_two_gib_disk(ram, disk):
    with pytest.raises(acquisition.ResourceAdmissionError):
        acquisition.enforce_resources({"available_physical_ram_bytes": ram, "free_disk_bytes": disk})


def test_exact_resource_floors_are_admitted():
    acquisition.enforce_resources({"available_physical_ram_bytes": acquisition.RAM_FLOOR,
                                   "free_disk_bytes": acquisition.DISK_FLOOR})


def test_pre_network_refusal_creates_receipt_without_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "ROOT", tmp_path)
    monkeypatch.setattr(acquisition, "PLAN", tmp_path / "plan.json")
    dataset = tmp_path / "data/external/court_source_wave64_v1"
    monkeypatch.setattr(acquisition, "DATASET", dataset)
    monkeypatch.setattr(acquisition, "load_frozen_plan", lambda: {})
    monkeypatch.setattr(acquisition, "current_resources", lambda _: {
        "available_physical_ram_bytes": acquisition.RAM_FLOOR - 1,
        "free_disk_bytes": acquisition.DISK_FLOOR + 1, "owned_rss_bytes": 1, "peak_owned_rss_bytes": 1})
    attempt = tmp_path / "attempt"
    receipt = acquisition.acquire(attempt)
    assert receipt["status"] == "REFUSED_RESOURCE_ADMISSION_BEFORE_NETWORK"
    assert receipt["requested_member_ranges"] == receipt["range_bytes_received"] == receipt["completed_images"] == 0
    assert receipt["dataset_exists"] is False
    assert not dataset.exists()
    assert json.loads((attempt / "receipt.json").read_text())["status"] == receipt["status"]
    with pytest.raises(FileExistsError, match="fresh immutable"):
        acquisition.acquire(attempt)


def test_existing_partial_dataset_is_preserved_without_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "ROOT", tmp_path)
    monkeypatch.setattr(acquisition, "PLAN", tmp_path / "plan.json")
    dataset = tmp_path / "data/external/court_source_wave64_v1"
    dataset.mkdir(parents=True)
    (dataset / "partial.png").write_bytes(b"preserve")
    monkeypatch.setattr(acquisition, "DATASET", dataset)
    with pytest.raises(FileExistsError, match="Preserve existing/partial"):
        acquisition.acquire(tmp_path / "attempt")
    assert (dataset / "partial.png").read_bytes() == b"preserve"
    assert not (tmp_path / "attempt").exists()


def test_workspace_path_cannot_escape_to_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        acquisition.within_workspace(tmp_path / ".." / "outside")
