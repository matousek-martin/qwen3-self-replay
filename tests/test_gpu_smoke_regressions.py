import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

from self_replay.paths import research


def test_single_gpu_rng_isolation_resolves_implicit_cuda_index():
    path = research() / "experiments/shorterbetter/config.py"
    spec = importlib.util.spec_from_file_location("shorterbetter_config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    current = Mock(return_value=0)
    assert config.rng_devices(SimpleNamespace(type="cuda", index=None), current) == [0]
    current.assert_called_once_with()
    current.reset_mock()
    assert config.rng_devices(SimpleNamespace(type="cuda", index=2), current) == [2]
    assert config.rng_devices(SimpleNamespace(type="cuda", index=0), current) == [0]
    assert config.rng_devices(SimpleNamespace(type="cpu", index=None), current) == []
    current.assert_not_called()
