# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Upstream logging for MoEvement localized recovery.

Logs intermediate activations and gradients at pipeline stage boundaries
during training. On failure, these logs enable localized recovery: only
the affected data-parallel group rolls back, using stored logs to replay
computations without requiring global rollback.
"""

from collections import defaultdict

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger


class LogEntry:
    """A single logged tensor at a pipeline stage boundary.

    Attributes:
        iteration: Training iteration number.
        micro_batch_id: Microbatch index within the iteration.
        stage_id: Pipeline stage that produced this tensor.
        direction: 'activation' (forward) or 'gradient' (backward).
        tensor: The logged tensor (stored on CPU).
    """

    def __init__(self, iteration, micro_batch_id, stage_id, direction, tensor):
        self.iteration = iteration
        self.micro_batch_id = micro_batch_id
        self.stage_id = stage_id
        self.direction = direction
        self.tensor = tensor

    def total_bytes(self):
        return self.tensor.nelement() * self.tensor.element_size()


class UpstreamLogger:
    """Manages logging of activations and gradients at pipeline stage boundaries.

    During training, copies tensors to pinned CPU memory using a dedicated CUDA
    stream. Logs are tagged with (iteration, microbatch_id, stage_id) for precise
    replay during recovery.
    """

    def __init__(self, max_window_iterations=10):
        self._logs = defaultdict(list)  # (iteration, micro_batch_id) -> [LogEntry]
        self._cuda_stream = None
        self._max_window = max_window_iterations
        self._current_iteration = -1
        self._oldest_iteration = -1

    def _get_cuda_stream(self):
        if self._cuda_stream is None:
            self._cuda_stream = get_accelerator().Stream()
        return self._cuda_stream

    def log_activation(self, tensor, iteration, micro_batch_id, stage_id):
        """Log an activation tensor being sent to the next pipeline stage.

        The tensor is asynchronously copied to pinned CPU memory.

        Args:
            tensor: The activation tensor on GPU.
            iteration: Current training iteration.
            micro_batch_id: Current microbatch index.
            stage_id: Pipeline stage sending the activation.
        """
        self._log_tensor(tensor, iteration, micro_batch_id, stage_id, "activation")

    def log_gradient(self, tensor, iteration, micro_batch_id, stage_id):
        """Log a gradient tensor being sent to the previous pipeline stage.

        Args:
            tensor: The gradient tensor on GPU.
            iteration: Current training iteration.
            micro_batch_id: Current microbatch index.
            stage_id: Pipeline stage sending the gradient.
        """
        self._log_tensor(tensor, iteration, micro_batch_id, stage_id, "gradient")

    def _log_tensor(self, tensor, iteration, micro_batch_id, stage_id, direction):
        """Internal method to copy a tensor to CPU and store as a log entry."""
        if tensor is None:
            return

        stream = self._get_cuda_stream()

        if isinstance(tensor, (tuple, list)):
            for i, t in enumerate(tensor):
                if t is not None and isinstance(t, torch.Tensor) and t.is_floating_point():
                    self._log_single_tensor(t, iteration, micro_batch_id, stage_id, f"{direction}_{i}", stream)
        elif isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            self._log_single_tensor(tensor, iteration, micro_batch_id, stage_id, direction, stream)

    def _log_single_tensor(self, tensor, iteration, micro_batch_id, stage_id, direction, stream):
        """Copy a single tensor to CPU asynchronously and record it."""
        with get_accelerator().stream(stream):
            cpu_tensor = torch.empty_like(tensor, device='cpu', pin_memory=True)
            cpu_tensor.copy_(tensor, non_blocking=True)

        entry = LogEntry(iteration=iteration,
                         micro_batch_id=micro_batch_id,
                         stage_id=stage_id,
                         direction=direction,
                         tensor=cpu_tensor)

        key = (iteration, micro_batch_id)
        self._logs[key].append(entry)
        self._current_iteration = max(self._current_iteration, iteration)

    def synchronize(self):
        """Wait for all async tensor copies to complete."""
        stream = self._get_cuda_stream()
        if stream is not None:
            stream.synchronize()

    def get_logs_for_iteration(self, iteration):
        """Retrieve all log entries for a specific iteration.

        Args:
            iteration: The iteration number.

        Returns:
            Dict of micro_batch_id -> list of LogEntry.
        """
        result = defaultdict(list)
        for (iter_num, mb_id), entries in self._logs.items():
            if iter_num == iteration:
                result[mb_id].extend(entries)
        return dict(result)

    def get_activations_for_replay(self, iteration, micro_batch_id, stage_id):
        """Get activation logs for replaying a specific microbatch at a stage.

        Args:
            iteration: Iteration to replay.
            micro_batch_id: Microbatch to replay.
            stage_id: Stage whose activations to retrieve.

        Returns:
            List of CPU tensors containing the logged activations.
        """
        key = (iteration, micro_batch_id)
        entries = self._logs.get(key, [])
        activations = []
        for entry in entries:
            if entry.stage_id == stage_id and entry.direction.startswith("activation"):
                activations.append(entry.tensor)
        return activations

    def get_gradients_for_replay(self, iteration, micro_batch_id, stage_id):
        """Get gradient logs for replaying a specific microbatch at a stage.

        Args:
            iteration: Iteration to replay.
            micro_batch_id: Microbatch to replay.
            stage_id: Stage whose gradients to retrieve.

        Returns:
            List of CPU tensors containing the logged gradients.
        """
        key = (iteration, micro_batch_id)
        entries = self._logs.get(key, [])
        gradients = []
        for entry in entries:
            if entry.stage_id == stage_id and entry.direction.startswith("gradient"):
                gradients.append(entry.tensor)
        return gradients

    def garbage_collect(self, oldest_valid_iteration):
        """Remove stale log entries from before the given iteration.

        Called after a new sparse checkpoint is persisted to free memory.

        Args:
            oldest_valid_iteration: Logs from before this iteration are removed.
        """
        stale_keys = [key for key in self._logs if key[0] < oldest_valid_iteration]
        freed_bytes = 0
        for key in stale_keys:
            for entry in self._logs[key]:
                freed_bytes += entry.total_bytes()
            del self._logs[key]

        if stale_keys:
            self._oldest_iteration = oldest_valid_iteration
            logger.info(f"[MoEvement] Garbage collected {len(stale_keys)} log entries "
                        f"(freed {freed_bytes / (1024**2):.1f} MB)")

    def total_memory_bytes(self):
        """Compute total CPU memory used by all stored logs."""
        total = 0
        for entries in self._logs.values():
            for entry in entries:
                total += entry.total_bytes()
        return total

    def clear(self):
        """Clear all log entries and free memory."""
        self._logs.clear()
        self._current_iteration = -1
        self._oldest_iteration = -1
