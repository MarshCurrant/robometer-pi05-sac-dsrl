"""Utilities for computing fingerprints/hashes of model parameters."""

import hashlib
from typing import Union


def fingerprint_bytes(data: Union[bytes, bytearray]) -> str:
    """
    Compute a fingerprint (hash) of binary data.

    Uses SHA256 to create a deterministic hash of the input bytes.
    Returns a hexadecimal string representation of the hash.

    Args:
        data: Binary data to fingerprint (bytes or bytearray)

    Returns:
        Hexadecimal string representation of the SHA256 hash (64 characters)

    Example:
        >>> fp = fingerprint_bytes(b"hello world")
        >>> len(fp) == 64
        True
    """
    if not data:
        return ""

    # Use SHA256 for a strong, deterministic hash
    hash_obj = hashlib.sha256(data)
    return hash_obj.hexdigest()
