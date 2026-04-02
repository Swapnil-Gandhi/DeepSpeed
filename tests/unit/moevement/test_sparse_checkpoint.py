# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for MoEvement sparse checkpointing system."""

import tempfile

import pytest
import torch

from deepspeed.moevement.config import MoEvementConfig, MOEVEMENT
from deepspeed.moevement.scheduler import (
    SparseCheckpointScheduler,
    OperatorInfo,
)
from deepspeed.moevement.sparse_snapshot import SparseSnapshotEngine
from deepspeed.moevement.conversion import SparseToDenseConverter
from deepspeed.moevement.upstream_logging import UpstreamLogger


class TestMoEvementConfig:

    def test_default_config(self):
        config = MoEvementConfig()
        assert config.enabled is False
        assert config.replication_factor == 2
        assert config.reorder_threshold == 0.10
        assert config.reorder_fraction == 0.25
        assert config.pcie_bandwidth_gbs == 25.0
        assert config.upstream_logging is True

    def test_config_from_dict(self):
        param_dict = {
            MOEVEMENT: {
                "enabled": True,
                "replication_factor": 3,
                "reorder_threshold": 0.05,
                "pcie_bandwidth_gbs": 32.0,
            }
        }
        config = MoEvementConfig(param_dict)
        assert config.enabled is True
        assert config.replication_factor == 3
        assert config.reorder_threshold == 0.05
        assert config.pcie_bandwidth_gbs == 32.0
        assert config.upstream_logging is True  # default

    def test_pcie_bandwidth_conversion(self):
        config = MoEvementConfig()
        expected = 25.0 * (1024**3)
        assert config.pcie_bandwidth_bytes_per_sec == expected


class TestSparseCheckpointScheduler:

    def _make_operators(self, num_experts=4, params_per_expert=1000, non_expert_params=500, gate_params=100):
        operators = []
        operators.append(
            OperatorInfo(name="non_expert",
                         num_params=non_expert_params,
                         is_expert=False,
                         layer_id=0,
                         local_expert_id=None))
        operators.append(
            OperatorInfo(name="gate_0", num_params=gate_params, is_expert=False, layer_id=0, local_expert_id=None))
        for i in range(num_experts):
            operators.append(
                OperatorInfo(name=f"expert_{i}",
                             num_params=params_per_expert,
                             is_expert=True,
                             layer_id=0,
                             local_expert_id=i))
        return operators

    def test_find_window_size_all_fit(self):
        """When all operators fit in a single iteration, W_sparse=1."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)  # Very fast PCIe
        operators = self._make_operators(num_experts=4, params_per_expert=1000)
        scheduler.register_operators(operators)

        w_sparse, num_active = scheduler.find_window_size(iter_time_sec=1.0)
        assert w_sparse == 1
        assert num_active == len(operators)

    def test_find_window_size_slow_pcie(self):
        """With slow PCIe, window size increases to spread checkpointing."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=100)  # Very slow
        operators = self._make_operators(num_experts=8, params_per_expert=100000)
        scheduler.register_operators(operators)

        w_sparse, num_active = scheduler.find_window_size(iter_time_sec=0.1)
        assert w_sparse > 1
        assert num_active < len(operators)

    def test_generate_schedule_covers_all_operators(self):
        """Schedule must snapshot every operator exactly once per window."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)
        operators = self._make_operators(num_experts=6)
        scheduler.register_operators(operators)

        w_sparse, schedule = scheduler.generate_schedule(iter_time_sec=1.0)
        assert len(schedule) == w_sparse

        # Collect all active operators across the window
        all_active = set()
        for entry in schedule:
            all_active.update(entry.active_operators)

        # Every operator must appear as active at least once
        for op in operators:
            assert op.name in all_active, f"Operator {op.name} not scheduled as active"

    def test_order_operators_popularity(self):
        """Operators are sorted: non-experts first, then experts by ascending activation count."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)
        operators = self._make_operators(num_experts=4)

        # Set different activation counts
        operators[2].activation_count = 100  # expert_0
        operators[3].activation_count = 50  # expert_1
        operators[4].activation_count = 200  # expert_2
        operators[5].activation_count = 10  # expert_3

        scheduler.register_operators(operators)
        ordered = scheduler.order_operators()

        # Non-experts come first
        assert not ordered[0].is_expert
        assert not ordered[1].is_expert

        # Experts sorted by ascending activation count
        expert_order = [op for op in ordered if op.is_expert]
        for i in range(len(expert_order) - 1):
            assert expert_order[i].activation_count <= expert_order[i + 1].activation_count

    def test_get_schedule_for_iteration(self):
        """Schedule cycles through window positions correctly."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=100)
        operators = self._make_operators(num_experts=8, params_per_expert=100000)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=0.1)

        # Each iteration should map to a specific schedule entry
        for step in range(20):
            entry = scheduler.get_schedule_for_iteration(step)
            assert entry is not None
            expected_idx = step % scheduler.w_sparse
            assert entry.active_operators == scheduler.schedule[expected_idx].active_operators

    def test_update_activation_counts(self):
        """Activation counts are accumulated correctly."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)
        operators = self._make_operators(num_experts=4)
        scheduler.register_operators(operators)

        counts = torch.tensor([10.0, 20.0, 30.0, 40.0])
        scheduler.update_activation_counts(layer_id=0, exp_counts=counts)

        expert_ops = [op for op in scheduler.operators if op.is_expert]
        assert expert_ops[0].activation_count == 10.0
        assert expert_ops[1].activation_count == 20.0
        assert expert_ops[2].activation_count == 30.0
        assert expert_ops[3].activation_count == 40.0

    def test_should_reorder_no_change(self):
        """No reorder when popularity hasn't changed."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12)
        operators = self._make_operators(num_experts=4)
        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=1.0)

        assert not scheduler.should_reorder()

    def test_should_reorder_after_significant_change(self):
        """Reorder triggers when enough experts shift position significantly."""
        scheduler = SparseCheckpointScheduler(pcie_bandwidth_bytes_per_sec=1e12,
                                              reorder_threshold=0.10,
                                              reorder_fraction=0.25)
        operators = self._make_operators(num_experts=8)

        # Initial ordering
        for i, op in enumerate(operators):
            if op.is_expert:
                op.activation_count = i * 10

        scheduler.register_operators(operators)
        scheduler.generate_schedule(iter_time_sec=1.0)

        # Reverse popularity ordering
        expert_ops = [op for op in scheduler.operators if op.is_expert]
        for i, op in enumerate(expert_ops):
            op.activation_count = (len(expert_ops) - i) * 1000

        assert scheduler.should_reorder()


class TestSparseSnapshotEngine:

    @pytest.fixture(autouse=True)
    def check_accelerator(self):
        """Skip tests that require CUDA stream if no accelerator is available."""
        try:
            from deepspeed.accelerator import get_accelerator
            accel = get_accelerator()
            if accel is None or accel.Stream is None:
                pytest.skip("No accelerator available for snapshot tests")
        except Exception:
            pytest.skip("No accelerator available for snapshot tests")

    def test_snapshot_active_operator(self):
        """Active operator snapshots FP32 weights and optimizer state."""
        engine = SparseSnapshotEngine(replication_factor=2)

        params = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        optim_state = {"exp_avg": torch.randn(10, 10), "exp_avg_sq": torch.randn(10, 10)}

        engine.snapshot_operator("expert_0", params, optim_state, is_active=True, iteration=0)
        engine.synchronize()

        snaps = engine.get_current_snapshots()
        assert "expert_0" in snaps
        snap = snaps["expert_0"]
        assert snap.is_active is True
        assert "params.weight" in snap.state_dict
        assert "optimizer.exp_avg" in snap.state_dict

    def test_snapshot_frozen_operator(self):
        """Frozen operator snapshots only FP16 compute weights."""
        engine = SparseSnapshotEngine(replication_factor=2)

        params = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}

        engine.snapshot_operator("expert_1", params, None, is_active=False, iteration=0)
        engine.synchronize()

        snaps = engine.get_current_snapshots()
        assert "expert_1" in snaps
        snap = snaps["expert_1"]
        assert snap.is_active is False
        assert "compute_weights.weight" in snap.state_dict
        # FP16 compute weights should be half precision
        assert snap.state_dict["compute_weights.weight"].dtype == torch.float16

    def test_save_and_load_disk(self):
        """Sparse snapshots can be saved to and loaded from disk."""
        engine = SparseSnapshotEngine(replication_factor=2)

        params = {"weight": torch.randn(5, 5)}
        engine.snapshot_operator("test_op", params, None, is_active=False, iteration=0)
        engine.synchronize()

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save_to_disk(tmpdir, "step_100")

            metadata, states = SparseSnapshotEngine.load_from_disk(tmpdir, "step_100")
            assert metadata is not None
            assert "test_op" in metadata["operator_names"]
            assert "test_op" in states


class TestSparseToDenseConverter:

    def test_initialize_from_snapshots(self):
        """Converter correctly classifies operators as active or frozen."""
        converter = SparseToDenseConverter()

        metadata = {
            "operator_names": ["expert_0", "expert_1", "non_expert"],
            "operator_active": {
                "expert_0": True,
                "expert_1": False,
                "non_expert": True
            },
            "operator_iterations": {
                "expert_0": 10,
                "expert_1": 10,
                "non_expert": 10
            },
        }
        states = {
            "expert_0": {
                "params.weight": torch.randn(5, 5),
                "optimizer.exp_avg": torch.randn(5, 5)
            },
            "expert_1": {
                "compute_weights.weight": torch.randn(5, 5).half()
            },
            "non_expert": {
                "params.weight": torch.randn(5, 5),
                "optimizer.exp_avg": torch.randn(5, 5)
            },
        }

        converter.initialize_from_snapshots(metadata, states, schedule=None)

        assert converter.is_operator_active("expert_0")
        assert converter.is_operator_frozen("expert_1")
        assert converter.is_operator_active("non_expert")
        assert not converter.is_conversion_complete()

    def test_activate_operators(self):
        """Operators transition from frozen to active correctly."""
        converter = SparseToDenseConverter()

        metadata = {
            "operator_names": ["expert_0", "expert_1"],
            "operator_active": {
                "expert_0": True,
                "expert_1": False
            },
            "operator_iterations": {
                "expert_0": 10,
                "expert_1": 10
            },
        }
        states = {
            "expert_0": {
                "params.weight": torch.randn(5, 5)
            },
            "expert_1": {
                "compute_weights.weight": torch.randn(5, 5).half()
            },
        }

        converter.initialize_from_snapshots(metadata, states, schedule=None)
        assert not converter.is_conversion_complete()

        # Activate expert_1
        converter.activate_operators(["expert_1"],
                                     fp32_weights={"expert_1": {
                                         "weight": torch.randn(5, 5)
                                     }},
                                     optimizer_states={"expert_1": {
                                         "exp_avg": torch.randn(5, 5)
                                     }})

        assert converter.is_operator_active("expert_1")
        assert converter.is_conversion_complete()

    def test_skip_weight_grad_for_frozen(self):
        """Frozen operators should skip weight gradient computation."""
        converter = SparseToDenseConverter()

        metadata = {
            "operator_names": ["expert_0", "expert_1"],
            "operator_active": {
                "expert_0": True,
                "expert_1": False
            },
            "operator_iterations": {
                "expert_0": 10,
                "expert_1": 10
            },
        }
        states = {
            "expert_0": {
                "params.weight": torch.randn(5, 5)
            },
            "expert_1": {
                "compute_weights.weight": torch.randn(5, 5).half()
            },
        }

        converter.initialize_from_snapshots(metadata, states, schedule=None)

        assert not converter.should_skip_weight_grad("expert_0")
        assert converter.should_skip_weight_grad("expert_1")
        assert not converter.should_skip_optimizer_step("expert_0")
        assert converter.should_skip_optimizer_step("expert_1")

    def test_replay_iterations(self):
        """Replay iteration tracking works correctly."""
        converter = SparseToDenseConverter()
        converter.set_replay_iterations([10, 11, 12])

        assert converter.get_remaining_replay_count() == 3
        assert converter.get_next_replay_iteration() == 10
        assert converter.get_next_replay_iteration() == 11
        assert converter.get_next_replay_iteration() == 12
        assert converter.get_next_replay_iteration() is None


class TestUpstreamLogger:

    @pytest.fixture(autouse=True)
    def check_accelerator(self):
        """Skip tests that require CUDA stream if no accelerator is available."""
        try:
            from deepspeed.accelerator import get_accelerator
            accel = get_accelerator()
            if accel is None or accel.Stream is None:
                pytest.skip("No accelerator available for upstream logging tests")
        except Exception:
            pytest.skip("No accelerator available for upstream logging tests")

    def test_log_activation(self):
        """Activations are logged with correct metadata."""
        logger = UpstreamLogger()

        tensor = torch.randn(4, 8)
        logger.log_activation(tensor, iteration=5, micro_batch_id=0, stage_id=1)
        logger.synchronize()

        logs = logger.get_logs_for_iteration(5)
        assert 0 in logs
        assert len(logs[0]) == 1
        assert logs[0][0].direction == "activation"
        assert logs[0][0].stage_id == 1

    def test_log_gradient(self):
        """Gradients are logged with correct metadata."""
        logger = UpstreamLogger()

        tensor = torch.randn(4, 8)
        logger.log_gradient(tensor, iteration=5, micro_batch_id=1, stage_id=2)
        logger.synchronize()

        logs = logger.get_logs_for_iteration(5)
        assert 1 in logs
        assert len(logs[1]) == 1
        assert logs[1][0].direction == "gradient"

    def test_get_activations_for_replay(self):
        """Specific activations can be retrieved for replay."""
        logger = UpstreamLogger()

        t1 = torch.randn(4, 8)
        t2 = torch.randn(4, 8)
        logger.log_activation(t1, iteration=5, micro_batch_id=0, stage_id=1)
        logger.log_gradient(t2, iteration=5, micro_batch_id=0, stage_id=1)
        logger.synchronize()

        acts = logger.get_activations_for_replay(5, 0, 1)
        assert len(acts) == 1

        grads = logger.get_gradients_for_replay(5, 0, 1)
        assert len(grads) == 1

    def test_garbage_collect(self):
        """Stale logs are properly garbage collected."""
        logger = UpstreamLogger()

        for i in range(10):
            logger.log_activation(torch.randn(4, 8), iteration=i, micro_batch_id=0, stage_id=0)
        logger.synchronize()

        assert logger.total_memory_bytes() > 0

        logger.garbage_collect(oldest_valid_iteration=7)

        # Logs for iterations 0-6 should be gone
        for i in range(7):
            assert len(logger.get_logs_for_iteration(i)) == 0

        # Logs for iterations 7-9 should still exist
        for i in range(7, 10):
            assert len(logger.get_logs_for_iteration(i)) > 0

    def test_log_tuple_tensors(self):
        """Tuple of tensors (common in pipeline) are logged correctly."""
        logger = UpstreamLogger()

        tensors = (torch.randn(4, 8), torch.randn(4, 8))
        logger.log_activation(tensors, iteration=1, micro_batch_id=0, stage_id=0)
        logger.synchronize()

        logs = logger.get_logs_for_iteration(1)
        assert 0 in logs
        assert len(logs[0]) == 2  # Two tensors from the tuple

    def test_clear(self):
        """Clear removes all logs."""
        logger = UpstreamLogger()
        logger.log_activation(torch.randn(4, 8), iteration=0, micro_batch_id=0, stage_id=0)
        logger.synchronize()

        logger.clear()
        assert logger.total_memory_bytes() == 0
