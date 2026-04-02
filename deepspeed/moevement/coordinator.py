# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""MoEvement coordinator: orchestrates sparse checkpointing, conversion, and logging.

The coordinator runs on each worker alongside DeepSpeed, managing the lifecycle
of sparse checkpoints, triggering snapshots each iteration, and coordinating
localized recovery on failure using upstream logs.
"""

from collections import OrderedDict

import torch

from deepspeed.utils import logger

from deepspeed.moevement.scheduler import SparseCheckpointScheduler, OperatorInfo
from deepspeed.moevement.sparse_snapshot import SparseSnapshotEngine
from deepspeed.moevement.conversion import SparseToDenseConverter
from deepspeed.moevement.upstream_logging import UpstreamLogger


class MoEvementCoordinator:
    """Orchestrates MoEvement's sparse checkpointing system.

    Manages the interaction between:
    - SparseCheckpointScheduler: determines what to checkpoint each iteration
    - SparseSnapshotEngine: handles GPU-to-CPU transfers and persistence
    - SparseToDenseConverter: reconstructs dense checkpoints during recovery
    - UpstreamLogger: logs activations/gradients for localized recovery
    """

    def __init__(self, config):
        """Initialize the MoEvement coordinator.

        Args:
            config: MoEvementConfig instance.
        """
        self.config = config
        self.scheduler = SparseCheckpointScheduler(
            pcie_bandwidth_bytes_per_sec=config.pcie_bandwidth_bytes_per_sec,
            reorder_threshold=config.reorder_threshold,
            reorder_fraction=config.reorder_fraction,
        )
        self.snapshot_engine = SparseSnapshotEngine(replication_factor=config.replication_factor)
        self.converter = SparseToDenseConverter()
        self.upstream_logger = UpstreamLogger() if config.upstream_logging else None

        self._initialized = False
        self._iter_time_sec = None
        self._global_step = 0
        self._window_step = 0  # step within current sparse window
        self._recovering = False
        self._moe_layers = []
        self._operator_map = OrderedDict()  # name -> (module, param_group_idx)

    def initialize(self, model, moe_layers, iter_time_sec):
        """Initialize the coordinator with model information.

        Discovers all operators (experts, non-experts, gating) and builds
        the initial checkpoint schedule.

        Args:
            model: The DeepSpeed model (nn.Module).
            moe_layers: List of MOELayer instances found in the model.
            iter_time_sec: Measured iteration time in seconds.
        """
        self._moe_layers = moe_layers
        self._iter_time_sec = iter_time_sec

        operators = self._discover_operators(model, moe_layers)
        self.scheduler.register_operators(operators)

        w_sparse, schedule = self.scheduler.generate_schedule(iter_time_sec)

        self._initialized = True
        logger.info(f"[MoEvement] Coordinator initialized: {len(operators)} operators, "
                    f"W_sparse={w_sparse}, iter_time={iter_time_sec:.3f}s")

    def _discover_operators(self, model, moe_layers):
        """Discover all checkpointable operators in the model.

        Operators include:
        - Each expert in each MoE layer
        - Non-expert (shared) layers
        - Gating networks

        Args:
            model: The model module.
            moe_layers: List of MOELayer instances.

        Returns:
            List of OperatorInfo objects.
        """
        operators = []

        # Discover non-MoE parameters as a single "non_expert" operator
        non_expert_params = 0
        for name, param in model.named_parameters():
            if not hasattr(param, 'allreduce') or param.allreduce:
                non_expert_params += param.numel()

        if non_expert_params > 0:
            op = OperatorInfo(name="non_expert",
                              num_params=non_expert_params,
                              is_expert=False,
                              layer_id=-1,
                              local_expert_id=None)
            operators.append(op)
            self._operator_map["non_expert"] = None

        # Discover experts and gating from each MoE layer
        for layer_idx, moe_layer in enumerate(moe_layers):
            # Gating operator
            gate_params = sum(p.numel() for p in moe_layer.gate.parameters())
            if gate_params > 0:
                gate_name = f"layer_{layer_idx}_gate"
                op = OperatorInfo(name=gate_name,
                                  num_params=gate_params,
                                  is_expert=False,
                                  layer_id=layer_idx,
                                  local_expert_id=None)
                operators.append(op)
                self._operator_map[gate_name] = moe_layer.gate

            # Expert operators
            num_local = moe_layer.num_local_experts
            for expert_idx in range(num_local):
                expert_module = moe_layer.experts.deepspeed_experts[expert_idx]
                expert_params = sum(p.numel() for p in expert_module.parameters())
                expert_name = f"layer_{layer_idx}_expert_{expert_idx}"
                op = OperatorInfo(name=expert_name,
                                  num_params=expert_params,
                                  is_expert=True,
                                  layer_id=layer_idx,
                                  local_expert_id=expert_idx)
                operators.append(op)
                self._operator_map[expert_name] = expert_module

        return operators

    def on_iteration_start(self, global_step):
        """Called at the beginning of each training iteration.

        Args:
            global_step: Current global training step.
        """
        self._global_step = global_step

    def on_forward_complete(self, moe_layer, layer_idx, exp_counts):
        """Called after a MoE layer's forward pass to track activation counts.

        Args:
            moe_layer: The MOELayer that completed forward.
            layer_idx: Index of this MoE layer.
            exp_counts: Tensor of expert activation counts.
        """
        self.scheduler.update_activation_counts(layer_idx, exp_counts)

    def on_iteration_end(self, global_step, model, optimizer):
        """Called after each training iteration to perform sparse checkpointing.

        Pulls expert activation counts from MoE layers, snapshots the scheduled
        subset of operators, and manages window progression and persistence.

        Args:
            global_step: Current global training step.
            model: The model module.
            optimizer: The optimizer.
        """
        if not self._initialized or self._recovering:
            return

        # Update expert activation counts from MoE layers
        for layer_idx, moe_layer in enumerate(self._moe_layers):
            if hasattr(moe_layer, 'exp_counts') and moe_layer.exp_counts is not None:
                self.scheduler.update_activation_counts(layer_idx, moe_layer.exp_counts)

        schedule_entry = self.scheduler.get_schedule_for_iteration(global_step)
        if schedule_entry is None:
            return

        # Snapshot active operators (FP32 master weights + optimizer state)
        for op_name in schedule_entry.active_operators:
            module = self._operator_map.get(op_name)
            if module is None and op_name == "non_expert":
                params_dict = self._get_non_expert_params(model)
                optim_state = self._get_non_expert_optimizer_state(model, optimizer)
            elif module is not None:
                params_dict = {name: p.data for name, p in module.named_parameters()}
                optim_state = self._get_module_optimizer_state(module, optimizer)
            else:
                continue

            self.snapshot_engine.snapshot_operator(name=op_name,
                                                   params_dict=params_dict,
                                                   optimizer_state_dict=optim_state,
                                                   is_active=True,
                                                   iteration=global_step)

        # Snapshot frozen operators (FP16 compute weights only)
        for op_name in schedule_entry.frozen_operators:
            module = self._operator_map.get(op_name)
            if module is None and op_name == "non_expert":
                params_dict = self._get_non_expert_params(model)
            elif module is not None:
                params_dict = {name: p.data for name, p in module.named_parameters()}
            else:
                continue

            self.snapshot_engine.snapshot_operator(name=op_name,
                                                   params_dict=params_dict,
                                                   optimizer_state_dict=None,
                                                   is_active=False,
                                                   iteration=global_step)

        # Manage window progression
        self._window_step += 1
        if self._window_step >= self.scheduler.w_sparse:
            self._window_step = 0
            self.snapshot_engine.synchronize()
            self.snapshot_engine.begin_window(global_step)

            # Garbage collect stale upstream logs
            if self.upstream_logger is not None:
                oldest_valid = global_step - self.scheduler.w_sparse
                self.upstream_logger.garbage_collect(oldest_valid)

            # Check if reordering is needed
            if self.scheduler.should_reorder():
                self.scheduler.generate_schedule(self._iter_time_sec)
                logger.info(f"[MoEvement] Reordered checkpoint schedule at step {global_step}")

    def on_send_activations(self, tensor, iteration, micro_batch_id, stage_id):
        """Hook called when pipeline stage sends activations downstream.

        Args:
            tensor: Activation tensor being sent.
            iteration: Current iteration.
            micro_batch_id: Current microbatch.
            stage_id: Sending stage ID.
        """
        if self.upstream_logger is not None:
            self.upstream_logger.log_activation(tensor, iteration, micro_batch_id, stage_id)

    def on_send_gradients(self, tensor, iteration, micro_batch_id, stage_id):
        """Hook called when pipeline stage sends gradients upstream.

        Args:
            tensor: Gradient tensor being sent.
            iteration: Current iteration.
            micro_batch_id: Current microbatch.
            stage_id: Sending stage ID.
        """
        if self.upstream_logger is not None:
            self.upstream_logger.log_gradient(tensor, iteration, micro_batch_id, stage_id)

    def save_sparse_checkpoint(self, save_dir, tag):
        """Save the current sparse checkpoint state to disk.

        Args:
            save_dir: Checkpoint directory.
            tag: Checkpoint tag.
        """
        self.snapshot_engine.synchronize()
        self.snapshot_engine.save_to_disk(save_dir, tag)

    def load_sparse_checkpoint(self, load_dir, tag, schedule=None):
        """Load sparse checkpoint and initialize conversion.

        Args:
            load_dir: Checkpoint directory.
            tag: Checkpoint tag.
            schedule: Optional checkpoint schedule to use.

        Returns:
            True if a sparse checkpoint was found and loaded.
        """
        metadata, operator_states = SparseSnapshotEngine.load_from_disk(load_dir, tag)
        if metadata is None:
            return False

        self.converter.initialize_from_snapshots(metadata, operator_states, schedule)
        self._recovering = True
        return True

    def begin_recovery(self, failed_stage_id=None, dp_group_rank=None):
        """Begin localized recovery after a failure.

        Only the affected data-parallel group rolls back. If upstream logging
        is enabled, recovery is confined to the failed pipeline stage.

        Args:
            failed_stage_id: The pipeline stage that failed (None for non-pipeline).
            dp_group_rank: The data-parallel group that needs recovery.
        """
        self._recovering = True

        if self.upstream_logger is not None and failed_stage_id is not None:
            logger.info(f"[MoEvement] Beginning localized recovery for stage {failed_stage_id}, "
                        f"DP group {dp_group_rank}")
        else:
            logger.info(f"[MoEvement] Beginning recovery (DP group {dp_group_rank})")

    def end_recovery(self):
        """Mark recovery as complete."""
        self._recovering = False
        self.converter.clear()
        logger.info("[MoEvement] Recovery complete")

    def is_recovering(self):
        """Check if the system is currently in recovery mode."""
        return self._recovering

    def is_conversion_complete(self):
        """Check if sparse-to-dense conversion is complete."""
        return self.converter.is_conversion_complete()

    def should_skip_weight_grad(self, operator_name):
        """Check if weight gradient should be skipped during conversion.

        Args:
            operator_name: Name of the operator.

        Returns:
            True if the operator is frozen and should skip weight gradients.
        """
        if not self._recovering:
            return False
        return self.converter.should_skip_weight_grad(operator_name)

    def should_skip_optimizer_step(self, operator_name):
        """Check if optimizer step should be skipped during conversion.

        Args:
            operator_name: Name of the operator.

        Returns:
            True if the operator is frozen and should skip optimizer step.
        """
        if not self._recovering:
            return False
        return self.converter.should_skip_optimizer_step(operator_name)

    def _get_non_expert_params(self, model):
        """Extract non-expert parameter dict from model."""
        params = {}
        for name, param in model.named_parameters():
            if not hasattr(param, 'allreduce') or param.allreduce:
                params[name] = param.data
        return params

    def _get_non_expert_optimizer_state(self, model, optimizer):
        """Extract optimizer state for non-expert parameters."""
        state_dict = {}
        for name, param in model.named_parameters():
            if not hasattr(param, 'allreduce') or param.allreduce:
                if param in optimizer.state:
                    for key, val in optimizer.state[param].items():
                        if isinstance(val, torch.Tensor):
                            state_dict[f"{name}.{key}"] = val
        return state_dict

    def _get_module_optimizer_state(self, module, optimizer):
        """Extract optimizer state for a specific module's parameters."""
        state_dict = {}
        for name, param in module.named_parameters():
            if param in optimizer.state:
                for key, val in optimizer.state[param].items():
                    if isinstance(val, torch.Tensor):
                        state_dict[f"{name}.{key}"] = val
        return state_dict

    def get_memory_usage(self):
        """Get memory usage breakdown for MoEvement components.

        Returns:
            Dict with memory usage in bytes for each component.
        """
        usage = {
            "snapshot_bytes": sum(s.total_bytes() for s in self.snapshot_engine.get_all_snapshots().values()),
        }
        if self.upstream_logger is not None:
            usage["upstream_log_bytes"] = self.upstream_logger.total_memory_bytes()
        return usage

    def cleanup(self):
        """Release all resources held by MoEvement."""
        self.snapshot_engine.clear()
        self.converter.clear()
        if self.upstream_logger is not None:
            self.upstream_logger.clear()
