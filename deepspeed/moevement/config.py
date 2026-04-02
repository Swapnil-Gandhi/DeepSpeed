# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

MOEVEMENT = "moevement"
MOEVEMENT_ENABLED = "enabled"
MOEVEMENT_ENABLED_DEFAULT = False
MOEVEMENT_REPLICATION_FACTOR = "replication_factor"
MOEVEMENT_REPLICATION_FACTOR_DEFAULT = 2
MOEVEMENT_REORDER_THRESHOLD = "reorder_threshold"
MOEVEMENT_REORDER_THRESHOLD_DEFAULT = 0.10
MOEVEMENT_REORDER_FRACTION = "reorder_fraction"
MOEVEMENT_REORDER_FRACTION_DEFAULT = 0.25
MOEVEMENT_PCIE_BANDWIDTH_GBS = "pcie_bandwidth_gbs"
MOEVEMENT_PCIE_BANDWIDTH_GBS_DEFAULT = 25.0
MOEVEMENT_UPSTREAM_LOGGING = "upstream_logging"
MOEVEMENT_UPSTREAM_LOGGING_DEFAULT = True


class MoEvementConfig:
    """Configuration for MoEvement sparse checkpointing system.

    Args:
        enabled: Whether MoEvement sparse checkpointing is enabled.
        replication_factor: Number of peer nodes to replicate sparse snapshots to.
        reorder_threshold: Fraction change in activation frequency that triggers reordering.
        reorder_fraction: Fraction of experts whose frequency must change to trigger reordering.
        pcie_bandwidth_gbs: Effective GPU-to-CPU PCIe bandwidth in GB/s for scheduling.
        upstream_logging: Whether to enable upstream logging for localized recovery.
    """

    def __init__(self, param_dict=None):
        if param_dict is None:
            param_dict = {}

        moevement_dict = param_dict.get(MOEVEMENT, {})

        self.enabled = moevement_dict.get(MOEVEMENT_ENABLED, MOEVEMENT_ENABLED_DEFAULT)
        self.replication_factor = moevement_dict.get(MOEVEMENT_REPLICATION_FACTOR,
                                                     MOEVEMENT_REPLICATION_FACTOR_DEFAULT)
        self.reorder_threshold = moevement_dict.get(MOEVEMENT_REORDER_THRESHOLD, MOEVEMENT_REORDER_THRESHOLD_DEFAULT)
        self.reorder_fraction = moevement_dict.get(MOEVEMENT_REORDER_FRACTION, MOEVEMENT_REORDER_FRACTION_DEFAULT)
        self.pcie_bandwidth_gbs = moevement_dict.get(MOEVEMENT_PCIE_BANDWIDTH_GBS,
                                                     MOEVEMENT_PCIE_BANDWIDTH_GBS_DEFAULT)
        self.upstream_logging = moevement_dict.get(MOEVEMENT_UPSTREAM_LOGGING, MOEVEMENT_UPSTREAM_LOGGING_DEFAULT)

    @property
    def pcie_bandwidth_bytes_per_sec(self):
        return self.pcie_bandwidth_gbs * (1024**3)


def get_moevement_config(param_dict):
    return MoEvementConfig(param_dict)
