"""Indexed projections of stable sequences without a duration-sized copy."""
from collections.abc import Sequence


class MappedSequenceView(Sequence):
    """Read one source item per request; slices retain only a ``range``.

    The source must stay open and stable throughout use. This is a view, not a
    snapshot or a checkpoint, and does not retain mapped item values in RAM.
    """

    __slots__ = ('_source', '_map', '_indices')

    def __init__(self, source, mapper, *, _indices=None):
        self._source = source
        self._map = mapper
        self._indices = range(len(source)) if _indices is None else _indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(self._source, self._map, _indices=self._indices[index])
        source_index = self._indices[index]
        return self._map(self._source[source_index], source_index)


def attribute_sequence(source, attribute, *, convert=None):
    """Keep historical list snapshots; project other stable sequences lazily."""
    def mapped(item, _index):
        value = getattr(item, attribute)
        return value if convert is None else convert(value)

    if isinstance(source, list):
        return [mapped(item, index) for index, item in enumerate(source)]
    return MappedSequenceView(source, mapped)
