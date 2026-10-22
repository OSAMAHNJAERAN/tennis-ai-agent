"""Temporary JSON registries retain complete parsing and the stored v1 codec."""
import json

import pytest

from src.api.result_storage import DiskSeries, IndexedJsonObject, ResultStorage


def test_indexed_root_registry_preserves_duplicates_order_and_tag_shaped_data(tmp_path):
    path = tmp_path / 'raw.json'
    path.write_text('{"calibrations":{"b":{"series":1},"a":{"object":2},"b":{"series":3}},'
                    '"nested":{"calibrations":{"ordinary":true}}}')
    with ResultStorage(tmp_path) as storage:
        ordinary = storage.read_json(path)
        raw = storage.read_json(path, indexed_root_objects=iter(['calibrations']))
        registry = raw['calibrations']
        assert isinstance(registry, IndexedJsonObject)
        assert list(registry) == ['b', 'a']
        assert dict(registry.items()) == ordinary['calibrations']
        assert isinstance(raw['nested']['calibrations'], dict)
        registry['b']['series'] = 100
        assert registry['b'] == {'series': 3}
        with pytest.raises(KeyError): registry[1]
        with pytest.raises(TypeError): registry[[]]


def test_large_registry_normalizes_without_a_new_published_storage_format(tmp_path):
    path = tmp_path / 'raw.json'
    path.write_text(json.dumps({'calibrations': {str(index): {'matrix': list(range(100))} for index in range(257)}}))
    destination = tmp_path / 'published.sqlite3'
    with ResultStorage(tmp_path) as storage:
        raw = storage.read_json(path, indexed_root_objects=('calibrations',))
        registry = raw['calibrations']
        assert isinstance(registry, IndexedJsonObject) and len(registry) == 257
        assert isinstance(registry['128']['matrix'], DiskSeries)
        storage.set_root({'previous': True})
        with pytest.raises(TypeError, match='not JSON serializable'):
            storage.set_root({'raw_registry': registry})
        assert storage.root() == {'previous': True}
        storage.set_root({'matrix': registry['128']['matrix']})
        storage.publish(destination)
    with ResultStorage.open(destination) as reader:
        assert list(reader.root()['matrix']) == list(range(100))
        assert reader._one('SELECT version FROM storage_format') == (1,)
        assert reader._one('SELECT COUNT(*) FROM series') == (1,)
    with pytest.raises(RuntimeError, match='not open'): len(registry)


@pytest.mark.parametrize('suffix', [' trailing', ',', ' {"second":1}'])
def test_indexed_objects_do_not_bypass_complete_syntax_and_eof(tmp_path, suffix):
    path = tmp_path / 'raw.json'
    path.write_text(json.dumps({'calibrations': {str(index): index for index in range(257)}}) + suffix)
    with ResultStorage(tmp_path) as storage:
        with pytest.raises(ValueError):
            storage.read_json(path, indexed_root_objects=('calibrations',))


def test_repeated_reads_have_separate_registry_indexes_and_closed_owner_guards(tmp_path):
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    first.write_text('{"calibrations":{"same":{"frame":1}}}')
    second.write_text('{"calibrations":{"same":{"frame":2}}}')
    with ResultStorage(tmp_path) as storage:
        one = storage.read_json(first, indexed_root_objects=('calibrations',))['calibrations']
        two = storage.read_json(second, indexed_root_objects=('calibrations',))['calibrations']
        assert one['same'] == {'frame': 1} and two['same'] == {'frame': 2}
        with pytest.raises(TypeError): storage.read_json(first, indexed_root_objects='calibrations')
        with pytest.raises(TypeError): storage.read_json(first, indexed_root_objects=(1,))
    for registry in (one, two):
        with pytest.raises(RuntimeError, match='not open'): registry['same']
