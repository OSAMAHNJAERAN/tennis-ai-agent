"""The metadata resolver must never grant general Python attribute lookup."""
import io
import pytest
import torch

from scripts.evaluate.export_racket_specialist_tensors import history_method_resolver


class ExampleHistory:
    def min(self):
        return 1

    def max(self):
        return 2

    def current(self):
        return 3

    def mean(self):
        return 4


class AttributeRequest:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def __reduce__(self):
        return getattr, (self.owner, self.name)


@pytest.mark.parametrize('name', ['min', 'max', 'current', 'mean'])
def test_standard_weights_only_resolves_only_reviewed_functions(name):
    data = io.BytesIO()
    torch.save(AttributeRequest(ExampleHistory, name), data)
    data.seek(0)
    original = list(torch.serialization.get_safe_globals())
    resolver = history_method_resolver(ExampleHistory)
    with torch.serialization.safe_globals([ExampleHistory, (resolver, 'builtins.getattr')]):
        function = torch.load(data, weights_only=True)
    assert function is ExampleHistory.__dict__[name]
    assert torch.serialization.get_safe_globals() == original


@pytest.mark.parametrize('name', ['__dict__', '__class__', '__subclasses__', '__globals__', 'statistics', 'update'])
def test_standard_loader_rejects_unreviewed_attribute_names(name):
    data = io.BytesIO()
    torch.save(AttributeRequest(ExampleHistory, name), data)
    data.seek(0)
    resolver = history_method_resolver(ExampleHistory)
    with torch.serialization.safe_globals([ExampleHistory, (resolver, 'builtins.getattr')]):
        with pytest.raises(ValueError, match='Only the four reviewed'):
            torch.load(data, weights_only=True)


def test_resolver_rejects_instances_subclasses_and_string_subclasses():
    class SubHistory(ExampleHistory):
        pass

    class SubString(str):
        pass

    resolver = history_method_resolver(ExampleHistory)
    for owner, name in [(ExampleHistory(), 'min'), (SubHistory, 'min'), (str, 'min'),
                        (ExampleHistory, SubString('min')), (ExampleHistory, 0)]:
        with pytest.raises(ValueError, match='Only the four reviewed'):
            resolver(owner, name)
