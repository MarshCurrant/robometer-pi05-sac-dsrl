import io
import zlib
import numpy as np
import torch


def ndarray_to_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    # Use np.save for shape/dtype-preserving binary
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def bytes_to_ndarray(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    buf.seek(0)
    return np.load(buf, allow_pickle=False)


def state_dict_to_bytes(state_dict: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    raw = buf.getvalue()
    return zlib.compress(raw, level=6)


def bytes_to_state_dict(data: bytes) -> dict:
    # Decompress if needed
    try:
        data = zlib.decompress(data)
    except zlib.error:
        # Not compressed; use as-is
        pass
    buf = io.BytesIO(data)
    buf.seek(0)
    return torch.load(buf, map_location="cpu")
