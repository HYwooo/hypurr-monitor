import pytest


@pytest.fixture
def sample_klines() -> list[list[int]]:
    """Sample K-line data for testing."""
    import time

    now = int(time.time() * 1000)
    return [
        [now - 4000 * 3600000, 65000, 66000, 64000, 65500, 100],
        [now - 3000 * 3600000, 65500, 66500, 65000, 66000, 110],
        [now - 2000 * 3600000, 66000, 67000, 65500, 66500, 120],
        [now - 1000 * 3600000, 66500, 68000, 66000, 67500, 130],
        [now, 67500, 68500, 67000, 68000, 140],
    ]
