import pytest

from gateway.config import Config


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.fixture
def clock():
    return FakeClock()


def make_config(**overrides) -> Config:
    base = {
        "providers": {
            "a": {"type": "mock"},
            "b": {"type": "mock"},
            "c": {"type": "mock"},
        },
        "routes": {
            "main": {"primary": [{"provider": "a", "model": "big"}],
                     "fallbacks": [{"provider": "b", "model": "small"}]},
            "lb": {"primary": [{"provider": "a", "model": "m", "weight": 3}, {"provider": "c", "model": "m", "weight": 1}]},
        },
        "pricing": {"big": {"input_per_mtok": 4.0, "output_per_mtok": 20.0},
                    "small": {"input_per_mtok": 0.1, "output_per_mtok": 0.5}},
        "keys": {"k1": {"name": "alice", "requests_per_minute": 1000, "tokens_per_minute": 1_000_000}},
        "retry": {"attempts": 2, "base_delay_s": 0.01, "max_delay_s": 0.05},
        "breaker": {"failure_threshold": 3, "cooldown_s": 30},
    }
    base.update(overrides)
    return Config.model_validate(base)
