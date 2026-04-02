# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Sparse checkpoint scheduling policy (Algorithm 1 from MoEvement paper).

Determines the sparse checkpoint window size (W_sparse) and generates
a per-iteration schedule of which operators are active (full FP32 snapshot)
vs frozen (FP16 compute weights only).
"""

import math

from deepspeed.utils import logger


class OperatorInfo:
    """Metadata for a single operator (expert, non-expert, or gating).

    Attributes:
        name: Unique identifier for this operator.
        num_params: Number of parameters in this operator.
        activation_count: Cumulative token activation count (for experts).
        is_expert: Whether this operator is an expert.
        layer_id: The layer index this operator belongs to.
        local_expert_id: Local expert index (None for non-expert/gating operators).
    """

    def __init__(self, name, num_params, is_expert=True, layer_id=0, local_expert_id=None):
        self.name = name
        self.num_params = num_params
        self.activation_count = 0
        self.is_expert = is_expert
        self.layer_id = layer_id
        self.local_expert_id = local_expert_id

    def __repr__(self):
        return f"OperatorInfo(name={self.name}, params={self.num_params}, act_count={self.activation_count})"


class CheckpointSchedule:
    """A single iteration's checkpoint assignment.

    Attributes:
        active_operators: List of operator names whose FP32 state is checkpointed.
        frozen_operators: List of operator names that only store FP16 compute weights.
    """

    def __init__(self, active_operators, frozen_operators):
        self.active_operators = active_operators
        self.frozen_operators = frozen_operators


class SparseCheckpointScheduler:
    """Implements Algorithm 1 from the MoEvement paper.

    Computes the sparse checkpoint window size and generates per-iteration
    schedules that assign operators to active or frozen states.
    """

    # Bytes per parameter for each state component (mixed-precision FP16-FP32 with Adam)
    FP32_MASTER_WEIGHT_BYTES = 4
    FP32_OPTIMIZER_STATE_BYTES = 8  # Adam: momentum (4) + variance (4)
    FP16_COMPUTE_WEIGHT_BYTES = 2

    def __init__(self, pcie_bandwidth_bytes_per_sec, reorder_threshold=0.10, reorder_fraction=0.25):
        self.pcie_bandwidth = pcie_bandwidth_bytes_per_sec
        self.reorder_threshold = reorder_threshold
        self.reorder_fraction = reorder_fraction
        self.operators = []
        self.w_sparse = 1
        self.schedule = []
        self._prev_popularity_order = None

    def register_operators(self, operators):
        """Register the list of operators to schedule.

        Args:
            operators: List of OperatorInfo objects.
        """
        self.operators = operators

    def find_window_size(self, iter_time_sec):
        """Determine the smallest W_sparse whose snapshot fits within one iteration.

        Starts with all operators active, then gradually transitions some to frozen
        until the estimated snapshot time fits within the iteration time.

        Args:
            iter_time_sec: Duration of a single training iteration in seconds.

        Returns:
            Tuple of (w_sparse, num_active_per_iter).
        """
        total_ops = len(self.operators)
        if total_ops == 0:
            return 1, 0

        num_active = total_ops
        s_compute = self.FP16_COMPUTE_WEIGHT_BYTES
        s_master = self.FP32_MASTER_WEIGHT_BYTES
        s_optim = self.FP32_OPTIMIZER_STATE_BYTES

        while num_active > 1:
            num_frozen = total_ops - num_active

            # Snapshot size: active operators get full FP32 state, frozen get FP16 only
            active_params = sum(op.num_params for op in self.operators[:num_active])
            frozen_params = sum(op.num_params for op in self.operators[num_active:])
            ckpt_size_bytes = (s_master + s_optim) * active_params + s_compute * frozen_params

            snapshot_time = ckpt_size_bytes / self.pcie_bandwidth
            if snapshot_time <= iter_time_sec:
                break

            num_active -= 1

        w_sparse = math.ceil(total_ops / num_active)
        return w_sparse, num_active

    def order_operators(self):
        """Sort operators by ascending activation frequency (popularity).

        Less popular experts are checkpointed first, keeping popular experts
        frozen longer during sparse-to-dense conversion to reduce recomputation cost.
        Non-expert operators (gating, non-expert layers) are placed first since they
        are always active and typically smaller.

        Returns:
            List of OperatorInfo sorted by ascending popularity.
        """
        non_experts = [op for op in self.operators if not op.is_expert]
        experts = [op for op in self.operators if op.is_expert]
        experts.sort(key=lambda op: op.activation_count)
        return non_experts + experts

    def generate_schedule(self, iter_time_sec):
        """Generate the full sparse checkpoint schedule.

        Args:
            iter_time_sec: Duration of a single training iteration in seconds.

        Returns:
            Tuple of (w_sparse, list of CheckpointSchedule for each iteration in the window).
        """
        ordered = self.order_operators()
        self.operators = ordered

        w_sparse, num_active = self.find_window_size(iter_time_sec)
        self.w_sparse = w_sparse

        schedule = []
        for i in range(w_sparse):
            start = i * num_active
            end = min(start + num_active, len(ordered))
            active_names = [op.name for op in ordered[start:end]]
            frozen_names = [op.name for op in ordered if op.name not in active_names]
            schedule.append(CheckpointSchedule(active_operators=active_names, frozen_operators=frozen_names))

        self.schedule = schedule
        self._prev_popularity_order = [op.name for op in ordered if op.is_expert]

        logger.info(f"[MoEvement] Sparse checkpoint schedule: W_sparse={w_sparse}, "
                    f"active_per_iter={num_active}, total_operators={len(ordered)}")

        return w_sparse, schedule

    def should_reorder(self):
        """Check if expert activation frequencies have changed enough to trigger reordering.

        Reordering is triggered when activation frequencies change by more than
        reorder_threshold for at least reorder_fraction of experts.

        Returns:
            True if reordering should be triggered.
        """
        if self._prev_popularity_order is None:
            return True

        experts = [op for op in self.operators if op.is_expert]
        if len(experts) == 0:
            return False

        current_order = sorted(experts, key=lambda op: op.activation_count)
        current_names = [op.name for op in current_order]

        if len(current_names) != len(self._prev_popularity_order):
            return True

        # Count how many experts changed position by more than threshold
        total_experts = len(current_names)
        changed_count = 0
        for i, name in enumerate(current_names):
            if name in self._prev_popularity_order:
                prev_idx = self._prev_popularity_order.index(name)
                relative_shift = abs(i - prev_idx) / total_experts
                if relative_shift > self.reorder_threshold:
                    changed_count += 1

        fraction_changed = changed_count / total_experts
        return fraction_changed >= self.reorder_fraction

    def update_activation_counts(self, layer_id, exp_counts):
        """Update activation counts for experts in a specific layer.

        Args:
            layer_id: The layer index.
            exp_counts: Tensor of shape [num_experts] with token counts per expert.
        """
        if exp_counts is None:
            return

        counts = exp_counts.detach().cpu()
        for op in self.operators:
            if op.is_expert and op.layer_id == layer_id and op.local_expert_id is not None:
                if op.local_expert_id < len(counts):
                    op.activation_count += counts[op.local_expert_id].item()

    def get_schedule_for_iteration(self, global_step):
        """Get the checkpoint schedule for a specific iteration.

        Args:
            global_step: The current training iteration number.

        Returns:
            CheckpointSchedule for this iteration, or None if no schedule exists.
        """
        if not self.schedule:
            return None
        idx = global_step % self.w_sparse
        return self.schedule[idx]
