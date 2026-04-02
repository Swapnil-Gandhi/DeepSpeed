# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Sparse snapshot engine for MoEvement.

Handles per-operator granularity GPU-to-CPU snapshots and asynchronous
replication to peer nodes. Active operators snapshot FP32 master weights
and optimizer state; frozen operators snapshot only FP16 compute weights.
"""

import os
from collections import OrderedDict

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger


class PinnedBuffer:
    """Manages a pinned CPU buffer for async GPU-to-CPU transfers.

    Allocates a pinned host buffer once and reuses it across iterations
    to avoid repeated allocation overhead.
    """

    def __init__(self, size_bytes, dtype=torch.uint8):
        self.size_bytes = size_bytes
        self.buffer = None
        self.dtype = dtype

    def ensure_allocated(self, size_bytes=None):
        if size_bytes is not None and size_bytes > self.size_bytes:
            self.size_bytes = size_bytes
            self.buffer = None

        if self.buffer is None:
            numel = self.size_bytes
            self.buffer = get_accelerator().pin_memory(torch.empty(numel, dtype=self.dtype))

    def get_slice(self, num_elements, dtype):
        """Get a view into the pinned buffer with the specified dtype."""
        byte_size = num_elements * dtype.itemsize if hasattr(
            dtype, 'itemsize') else num_elements * torch.finfo(dtype).bits // 8
        self.ensure_allocated(byte_size)
        return self.buffer[:byte_size].view(dtype)[:num_elements]


class OperatorSnapshot:
    """Stores a snapshot of a single operator's state.

    For active operators: contains FP32 master weights and optimizer state.
    For frozen operators: contains only FP16 compute weights.
    """

    def __init__(self, name, iteration, is_active):
        self.name = name
        self.iteration = iteration
        self.is_active = is_active
        self.state_dict = {}

    def add_tensor(self, key, tensor):
        """Store a tensor (already on CPU) in this snapshot."""
        self.state_dict[key] = tensor

    def total_bytes(self):
        total = 0
        for t in self.state_dict.values():
            total += t.nelement() * t.element_size()
        return total


class SparseSnapshotEngine:
    """Manages per-operator sparse snapshots with async GPU-to-CPU transfers.

    Each iteration, only a scheduled subset of operators have their full FP32
    state snapshotted. The rest store only FP16 compute weights.
    """

    def __init__(self, replication_factor=2):
        self.replication_factor = replication_factor
        self._cuda_stream = None
        self._snapshots = OrderedDict()  # name -> OperatorSnapshot
        self._persisted_snapshots = OrderedDict()
        self._in_flight_snapshots = OrderedDict()
        self._snapshot_iteration = -1
        self._window_start_iteration = -1

    def _get_cuda_stream(self):
        if self._cuda_stream is None:
            self._cuda_stream = get_accelerator().Stream()
        return self._cuda_stream

    def snapshot_operator(self, name, params_dict, optimizer_state_dict, is_active, iteration):
        """Snapshot a single operator's state from GPU to CPU.

        For active operators, captures FP32 master weights and optimizer state.
        For frozen operators, captures only FP16 compute weights.

        Args:
            name: Operator identifier.
            params_dict: Dict of parameter name -> GPU tensor.
            optimizer_state_dict: Dict of optimizer state key -> GPU tensor (or None for frozen).
            is_active: Whether this operator is active (full FP32) or frozen (FP16 only).
            iteration: Current training iteration number.
        """
        snap = OperatorSnapshot(name, iteration, is_active)
        stream = self._get_cuda_stream()

        with get_accelerator().stream(stream):
            if is_active:
                # Snapshot FP32 master weights and optimizer state
                for key, tensor in params_dict.items():
                    cpu_tensor = torch.empty_like(tensor, device='cpu', pin_memory=True)
                    cpu_tensor.copy_(tensor, non_blocking=True)
                    snap.add_tensor(f"params.{key}", cpu_tensor)

                if optimizer_state_dict is not None:
                    for key, tensor in optimizer_state_dict.items():
                        if isinstance(tensor, torch.Tensor):
                            cpu_tensor = torch.empty_like(tensor, device='cpu', pin_memory=True)
                            cpu_tensor.copy_(tensor, non_blocking=True)
                            snap.add_tensor(f"optimizer.{key}", cpu_tensor)
            else:
                # Snapshot only FP16 compute weights
                for key, tensor in params_dict.items():
                    fp16_tensor = tensor.half() if tensor.dtype != torch.float16 else tensor
                    cpu_tensor = torch.empty_like(fp16_tensor, device='cpu', pin_memory=True)
                    cpu_tensor.copy_(fp16_tensor, non_blocking=True)
                    snap.add_tensor(f"compute_weights.{key}", cpu_tensor)

        self._snapshots[name] = snap

    def synchronize(self):
        """Wait for all async GPU-to-CPU transfers to complete."""
        stream = self._get_cuda_stream()
        stream.synchronize()

    def begin_window(self, iteration):
        """Mark the beginning of a new sparse checkpoint window."""
        self._window_start_iteration = iteration
        self._in_flight_snapshots = OrderedDict(self._snapshots)
        self._snapshots = OrderedDict()

    def finalize_window(self):
        """Finalize the current window: promote in-flight to persisted, GC old."""
        self._persisted_snapshots = OrderedDict(self._in_flight_snapshots)
        self._in_flight_snapshots = OrderedDict()

    def replicate_to_peers(self, process_group, peer_ranks):
        """Asynchronously replicate snapshots to peer nodes.

        Uses deepspeed.comm send/recv for point-to-point replication.

        Args:
            process_group: The distributed process group.
            peer_ranks: List of peer rank IDs to replicate to.
        """
        if not peer_ranks:
            return

        self.synchronize()

        for snap in self._in_flight_snapshots.values():
            for key, tensor in snap.state_dict.items():
                for peer_rank in peer_ranks[:self.replication_factor]:
                    dist.send(tensor, dst=peer_rank, group=process_group)

    def get_persisted_snapshots(self):
        """Get the most recently persisted sparse checkpoint snapshots.

        Returns:
            OrderedDict of operator name -> OperatorSnapshot.
        """
        return self._persisted_snapshots

    def get_current_snapshots(self):
        """Get the current in-progress sparse snapshots.

        Returns:
            OrderedDict of operator name -> OperatorSnapshot.
        """
        return self._snapshots

    def get_all_snapshots(self):
        """Get all snapshots (persisted + in-flight + current).

        Returns:
            OrderedDict of operator name -> OperatorSnapshot.
        """
        all_snaps = OrderedDict()
        all_snaps.update(self._persisted_snapshots)
        all_snaps.update(self._in_flight_snapshots)
        all_snaps.update(self._snapshots)
        return all_snaps

    def clear(self):
        """Clear all snapshots and free memory."""
        self._snapshots.clear()
        self._persisted_snapshots.clear()
        self._in_flight_snapshots.clear()

    def save_to_disk(self, save_dir, tag):
        """Persist sparse snapshots to disk for durable storage.

        Args:
            save_dir: Directory to save checkpoint files.
            tag: Checkpoint tag (e.g., global_step number).
        """
        checkpoint_dir = os.path.join(save_dir, str(tag), "moevement")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.synchronize()

        all_snaps = self.get_all_snapshots()
        metadata = {
            "window_start_iteration": self._window_start_iteration,
            "operator_names": list(all_snaps.keys()),
            "operator_active": {
                name: snap.is_active
                for name, snap in all_snaps.items()
            },
            "operator_iterations": {
                name: snap.iteration
                for name, snap in all_snaps.items()
            },
        }

        torch.save(metadata, os.path.join(checkpoint_dir, "metadata.pt"))

        for name, snap in all_snaps.items():
            safe_name = name.replace("/", "_").replace(".", "_")
            torch.save(snap.state_dict, os.path.join(checkpoint_dir, f"{safe_name}.pt"))

        logger.info(f"[MoEvement] Saved sparse checkpoint to {checkpoint_dir} "
                    f"with {len(all_snaps)} operator snapshots")

    @staticmethod
    def load_from_disk(load_dir, tag):
        """Load sparse snapshots from disk.

        Args:
            load_dir: Directory containing checkpoint files.
            tag: Checkpoint tag.

        Returns:
            Tuple of (metadata dict, dict of operator name -> state_dict).
        """
        checkpoint_dir = os.path.join(load_dir, str(tag), "moevement")
        if not os.path.exists(checkpoint_dir):
            return None, None

        metadata = torch.load(os.path.join(checkpoint_dir, "metadata.pt"), weights_only=False)

        operator_states = {}
        for name in metadata["operator_names"]:
            safe_name = name.replace("/", "_").replace(".", "_")
            filepath = os.path.join(checkpoint_dir, f"{safe_name}.pt")
            if os.path.exists(filepath):
                operator_states[name] = torch.load(filepath, weights_only=False)

        logger.info(f"[MoEvement] Loaded sparse checkpoint from {checkpoint_dir} "
                    f"with {len(operator_states)} operator snapshots")

        return metadata, operator_states
