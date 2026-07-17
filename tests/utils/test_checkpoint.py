# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.utils import checkpoint as checkpoint_module
from jaqmc.utils.checkpoint import NumPyCheckpointManager


def test_checkpoint_save_is_atomic_and_cleans_failed_temporary_file(
    tmp_path,
    monkeypatch,
):
    manager = NumPyCheckpointManager(tmp_path, prefix="state")
    fallback = {"value": jnp.asarray([0.0])}
    first = {"value": jnp.asarray([1.0])}
    first_path = manager.save(0, first)
    first_bytes = first_path.read_bytes()

    def fail_after_partial_write(file_obj, **_payload):
        file_obj.write(b"partial")
        raise RuntimeError("injected checkpoint failure")

    monkeypatch.setattr(
        checkpoint_module.np,
        "savez_compressed",
        fail_after_partial_write,
    )

    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        manager.save(1, {"value": jnp.asarray([2.0])})

    assert first_path.read_bytes() == first_bytes
    assert not (tmp_path / "state_ckpt_000001.npz").exists()
    assert not list(tmp_path.glob(".*.tmp"))
    step, restored = manager.restore(fallback)
    assert step == 1
    np.testing.assert_allclose(np.asarray(restored["value"]), [1.0])


def test_checkpoint_restore_falls_back_from_truncated_latest_file(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="state")
    fallback = {"value": jnp.asarray([0.0])}
    manager.save(0, {"value": jnp.asarray([1.0])})
    latest = manager.save(1, {"value": jnp.asarray([2.0])})
    latest.write_bytes(b"truncated")

    step, restored = manager.restore(fallback)

    assert step == 1
    np.testing.assert_allclose(np.asarray(restored["value"]), [1.0])


def test_checkpoint_restore_falls_back_when_latest_tree_is_incomplete(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="state")
    fallback = {"value": jnp.asarray([0.0])}
    manager.save(0, {"value": jnp.asarray([1.0])})
    latest = manager.save(1, {"value": jnp.asarray([2.0])})
    with latest.open("wb") as f_out:
        np.savez(f_out, step=1, unrelated=np.asarray([2.0]))

    step, restored = manager.restore(fallback)

    assert step == 1
    np.testing.assert_allclose(np.asarray(restored["value"]), [1.0])


def test_checkpoint_npz_does_not_store_allow_pickle_as_payload(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="state")
    path = manager.save(0, {"value": jnp.asarray([1.0])})

    with path.open("rb") as f_in, np.load(f_in, allow_pickle=False) as npf:
        assert "allow_pickle" not in npf.files
