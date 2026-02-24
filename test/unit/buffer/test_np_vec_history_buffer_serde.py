"""
Test default serialization and deserialization functions
"""

import pytest
import numpy as np
import torch

from frankenstein.buffer.vec_history import make_default_serde


def test_numpy_int8():
    ser_fn, _, deser_fn = make_default_serde()

    rng = np.random.default_rng()
    arr = rng.integers(0, 10, size=(3, 4), dtype=np.int8)

    serialized = ser_fn.obs(arr)
    deserialized = deser_fn.obs(serialized)

    assert (deserialized == arr).all()


def test_numpy_int64():
    ser_fn, _, deser_fn = make_default_serde()

    rng = np.random.default_rng()
    arr = rng.integers(0, 10, size=(3, 4), dtype=np.int64)

    serialized = ser_fn.obs(arr)
    deserialized = deser_fn.obs(serialized)

    assert (deserialized == arr).all()


def test_numpy_float64():
    ser_fn, _, deser_fn = make_default_serde()

    rng = np.random.default_rng()
    arr = rng.random(size=(3, 4), dtype=np.float64)

    serialized = ser_fn.obs(arr)
    deserialized = deser_fn.obs(serialized)

    assert (deserialized == arr).all()


def test_torch_float32():
    ser_fn, _, deser_fn = make_default_serde()

    arr = torch.rand((3, 4), dtype=torch.float32)

    serialized = ser_fn.obs(arr)
    deserialized = deser_fn.obs(serialized)

    assert (deserialized == arr).all()
