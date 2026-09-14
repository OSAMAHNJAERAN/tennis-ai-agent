import io
import zipfile

import pytest

from scripts.data.acquire_uvy_tennis import decode_member, safe_member


def fixture():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('UVY/tennis_V01/gt/gt.txt', b'1,1,20,30,40,50,1,1,1\n')
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
        info = archive.infolist()[0]
        item = {'name': info.filename, 'offset': info.header_offset + 1000,
                'compressed_bytes': info.compress_size, 'bytes': info.file_size,
                'compression': info.compress_type, 'crc32': info.CRC}
    return stream.getvalue(), item


def test_exact_member_decode_checks_header_size_and_crc():
    data, item = fixture()
    assert decode_member(data, 1000, item) == b'1,1,20,30,40,50,1,1,1\n'
    item['crc32'] += 1
    with pytest.raises(ValueError, match='CRC'):
        decode_member(data, 1000, item)


@pytest.mark.parametrize('change', ['name', 'size', 'offset', 'truncated'])
def test_inconsistent_archive_member_cannot_be_extracted(change):
    data, item = fixture()
    if change == 'name':
        item['name'] = 'UVY/tennis_V02/gt/gt.txt'
    elif change == 'size':
        item['bytes'] = 1
    elif change == 'offset':
        item['offset'] = 999
    else:
        data = data[:32]
    with pytest.raises((ValueError, UnicodeDecodeError)):
        decode_member(data, 1000, item)


@pytest.mark.parametrize('name', ['/UVY/tennis_V01/gt/gt.txt', '../gt.txt',
                                  'UVY/tennis_V01/../outside.txt', 'UVY\\tennis_V01\\gt.txt'])
def test_unsafe_paths_are_rejected(name):
    with pytest.raises(ValueError):
        safe_member(name)


def test_only_expected_tennis_data_members_are_selected():
    assert safe_member('UVY/tennis_V03/img1/000001.jpg')
    assert safe_member('UVY/tennis_V02/vid_ab-C_01.mp4')
    assert not safe_member('UVY/tennis_V01/desktop.ini')
    assert not safe_member('UVY/soccer_V01/gt/gt.txt')
