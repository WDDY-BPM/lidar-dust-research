"""Supply FPS starts as graph inputs without capturing or changing CPU RNG draws."""

from contextlib import contextmanager
from contextvars import ContextVar


_STARTS = ContextVar("fps_starts", default=None)


@contextmanager
def supplied_fps_starts(starts):
    """Consume exactly one (input point count, start tensor) per FPS layer."""
    iterator = iter(starts)
    token = _STARTS.set(iterator)
    try:
        yield
        if next(iterator, None) is not None:
            raise RuntimeError("Model consumed fewer FPS starts than expected")
    finally:
        _STARTS.reset(token)


def supplied_fps_start(batch_size, point_count, device):
    iterator = _STARTS.get()
    if iterator is None:
        return None
    try:
        expected_count, start = next(iterator)
    except StopIteration as error:
        raise RuntimeError("Model consumed more FPS starts than expected") from error
    if expected_count != point_count or start.shape != (batch_size,) or start.device != device:
        raise RuntimeError("FPS start shape/device does not match the captured model")
    return start
