"""A response must retain one coherent saved generation until it closes."""

from contextlib import closing
import sqlite3

import pytest

from src.api.result_storage import ResultStorage


def published_series(tmp_path):
    with ResultStorage(tmp_path) as storage:
        storage.set_root({'frames': storage.series([{'frame': 0}, {'frame': 1}])})
        return storage.publish(tmp_path / 'snapshot.sqlite3')


def test_one_reader_keeps_generation_snapshot_across_external_atomic_commit(tmp_path):
    destination = published_series(tmp_path)
    # WAL lets the external writer commit while a reader owns its snapshot.
    # Published generations are immutable in production; this deliberately
    # violates that contract to catch a response that mixes generations.
    with closing(sqlite3.connect(destination, isolation_level=None)) as writer:
        with closing(writer.execute('PRAGMA journal_mode=WAL')) as cursor:
            assert cursor.fetchone() == ('wal',)
        with ResultStorage.open(destination) as reader:
            frames = reader.root()['frames']
            assert frames[0] == {'frame': 0}
            with closing(writer.execute(
                'UPDATE items SET payload=(SELECT payload FROM items WHERE position=0) WHERE position=1'
            )):
                pass
            # The already opened response must keep the original frame, even
            # though another connection has committed changed database bytes.
            assert frames[1] == {'frame': 1}
            assert list(reader.root()['frames']) == [{'frame': 0}, {'frame': 1}]
        with ResultStorage.open(destination) as later_reader:
            assert list(later_reader.root()['frames']) == [{'frame': 0}, {'frame': 0}]


@pytest.mark.parametrize('body_failure', [False, True])
def test_reader_releases_snapshot_after_success_or_failed_response(tmp_path, body_failure):
    destination = published_series(tmp_path)
    reader = ResultStorage.open(destination)
    try:
        with reader:
            frames = reader.root()['frames']
            iterator = iter(frames)
            assert next(iterator) == {'frame': 0}
            if body_failure:
                raise RuntimeError('response failed')
    except RuntimeError as error:
        assert body_failure and str(error) == 'response failed'
    # Default rollback-journal publication must be writable immediately after
    # cleanup, proving that neither an unfinished iterator nor an exception
    # leaves a response lock behind.
    with closing(sqlite3.connect(destination, timeout=0)) as writer:
        with closing(writer.execute('UPDATE series SET length=length')):
            pass
        writer.commit()
    with pytest.raises(RuntimeError, match='not open'):
        _ = frames[0]
    with pytest.raises(RuntimeError, match='not open'):
        next(iterator)


def test_failed_root_validation_releases_reader_snapshot(tmp_path):
    destination = published_series(tmp_path)
    with closing(sqlite3.connect(destination)) as writer:
        with closing(writer.execute('DELETE FROM root')):
            pass
        writer.commit()
    with pytest.raises(ValueError, match='no completed root'):
        with ResultStorage.open(destination):
            pytest.fail('An incomplete root must not open a response')
    with closing(sqlite3.connect(destination, timeout=0)) as writer:
        with closing(writer.execute('UPDATE series SET length=length')):
            pass
        writer.commit()
