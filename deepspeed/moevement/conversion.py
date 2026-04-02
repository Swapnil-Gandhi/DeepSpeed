# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Sparse-to-dense checkpoint conversion for MoEvement.

Incrementally reconstructs a logically consistent dense checkpoint from
multiple sparse snapshots by replaying microbatches and transitioning
operators from frozen to active state as their FP32 state becomes available.
"""

from collections import OrderedDict
from enum import Enum

from deepspeed.utils import logger


class OperatorState(Enum):
    """Execution state of an operator during sparse-to-dense conversion."""
    FROZEN = "frozen"
    ACTIVE = "active"


class SparseToDenseConverter:
    """Manages the incremental reconstruction of a dense checkpoint.

    During recovery, operators are loaded from sparse snapshots in schedule
    order. Active operators (FP32 available) perform full forward + backward +
    optimizer updates. Frozen operators (FP16 only) perform forward and
    input-gradient computations but skip weight-gradient and optimizer updates.

    The conversion is complete when all operators are in the ACTIVE state.
    """

    def __init__(self):
        self._operator_states = OrderedDict()  # name -> OperatorState
        self._operator_fp32_weights = {}  # name -> dict of param tensors
        self._operator_optimizer_states = {}  # name -> dict of optimizer state tensors
        self._operator_fp16_weights = {}  # name -> dict of param tensors
        self._conversion_complete = False
        self._replay_iterations = []  # List of iteration numbers to replay
        self._current_replay_idx = 0

    def initialize_from_snapshots(self, metadata, operator_states, schedule):
        """Initialize operator states from loaded sparse snapshots.

        Operators are marked ACTIVE if their FP32 master weights and optimizer
        state are available, otherwise FROZEN.

        Args:
            metadata: Metadata dict from sparse checkpoint.
            operator_states: Dict of operator name -> state_dict.
            schedule: List of CheckpointSchedule objects defining the window.
        """
        self._operator_states.clear()
        self._operator_fp32_weights.clear()
        self._operator_optimizer_states.clear()
        self._operator_fp16_weights.clear()

        for name in metadata["operator_names"]:
            is_active = metadata["operator_active"].get(name, False)
            state_dict = operator_states.get(name, {})

            if is_active:
                self._operator_states[name] = OperatorState.ACTIVE
                self._operator_fp32_weights[name] = {
                    k.replace("params.", ""): v
                    for k, v in state_dict.items() if k.startswith("params.")
                }
                self._operator_optimizer_states[name] = {
                    k.replace("optimizer.", ""): v
                    for k, v in state_dict.items() if k.startswith("optimizer.")
                }
            else:
                self._operator_states[name] = OperatorState.FROZEN
                self._operator_fp16_weights[name] = {
                    k.replace("compute_weights.", ""): v
                    for k, v in state_dict.items() if k.startswith("compute_weights.")
                }

        self._conversion_complete = all(s == OperatorState.ACTIVE for s in self._operator_states.values())

        active_count = sum(1 for s in self._operator_states.values() if s == OperatorState.ACTIVE)
        frozen_count = sum(1 for s in self._operator_states.values() if s == OperatorState.FROZEN)
        logger.info(f"[MoEvement] Initialized conversion: {active_count} active, "
                    f"{frozen_count} frozen operators")

    def activate_operators(self, operator_names, fp32_weights, optimizer_states):
        """Transition operators from FROZEN to ACTIVE.

        Called when loading subsequent sparse snapshots during conversion.

        Args:
            operator_names: List of operator names to activate.
            fp32_weights: Dict of operator name -> param dict.
            optimizer_states: Dict of operator name -> optimizer state dict.
        """
        for name in operator_names:
            if name in self._operator_states:
                self._operator_states[name] = OperatorState.ACTIVE
                if name in fp32_weights:
                    self._operator_fp32_weights[name] = fp32_weights[name]
                if name in optimizer_states:
                    self._operator_optimizer_states[name] = optimizer_states[name]
                # Remove FP16 weights since FP32 is now available
                self._operator_fp16_weights.pop(name, None)

        self._conversion_complete = all(s == OperatorState.ACTIVE for s in self._operator_states.values())

        if self._conversion_complete:
            logger.info("[MoEvement] Sparse-to-dense conversion complete: all operators active")

    def is_operator_active(self, name):
        """Check if an operator is in the ACTIVE state.

        Args:
            name: Operator identifier.

        Returns:
            True if the operator is active, False if frozen.
        """
        return self._operator_states.get(name, OperatorState.FROZEN) == OperatorState.ACTIVE

    def is_operator_frozen(self, name):
        """Check if an operator is in the FROZEN state."""
        return self._operator_states.get(name, OperatorState.FROZEN) == OperatorState.FROZEN

    def is_conversion_complete(self):
        """Check if all operators have been transitioned to ACTIVE state."""
        return self._conversion_complete

    def get_operator_state(self, name):
        """Get the current state of an operator."""
        return self._operator_states.get(name, OperatorState.FROZEN)

    def get_active_operators(self):
        """Get names of all active operators."""
        return [name for name, state in self._operator_states.items() if state == OperatorState.ACTIVE]

    def get_frozen_operators(self):
        """Get names of all frozen operators."""
        return [name for name, state in self._operator_states.items() if state == OperatorState.FROZEN]

    def get_fp32_weights(self, name):
        """Get FP32 master weights for an active operator."""
        return self._operator_fp32_weights.get(name)

    def get_optimizer_state(self, name):
        """Get optimizer state for an active operator."""
        return self._operator_optimizer_states.get(name)

    def get_fp16_weights(self, name):
        """Get FP16 compute weights for a frozen operator."""
        return self._operator_fp16_weights.get(name)

    def set_replay_iterations(self, iterations):
        """Set the list of iterations that need to be replayed.

        Args:
            iterations: List of iteration numbers in order.
        """
        self._replay_iterations = list(iterations)
        self._current_replay_idx = 0

    def get_next_replay_iteration(self):
        """Get the next iteration number to replay.

        Returns:
            The next iteration number, or None if replay is complete.
        """
        if self._current_replay_idx >= len(self._replay_iterations):
            return None
        iteration = self._replay_iterations[self._current_replay_idx]
        self._current_replay_idx += 1
        return iteration

    def get_remaining_replay_count(self):
        """Get the number of iterations still to replay."""
        return len(self._replay_iterations) - self._current_replay_idx

    def should_skip_weight_grad(self, operator_name):
        """Check if weight gradient computation should be skipped for an operator.

        Frozen operators skip weight-gradient computation and optimizer updates,
        performing only forward and input-gradient computations.

        Args:
            operator_name: The operator to check.

        Returns:
            True if weight gradient should be skipped (operator is frozen).
        """
        return self.is_operator_frozen(operator_name)

    def should_skip_optimizer_step(self, operator_name):
        """Check if optimizer step should be skipped for an operator.

        Args:
            operator_name: The operator to check.

        Returns:
            True if optimizer step should be skipped (operator is frozen).
        """
        return self.is_operator_frozen(operator_name)

    def clear(self):
        """Reset all conversion state."""
        self._operator_states.clear()
        self._operator_fp32_weights.clear()
        self._operator_optimizer_states.clear()
        self._operator_fp16_weights.clear()
        self._conversion_complete = False
        self._replay_iterations = []
        self._current_replay_idx = 0
