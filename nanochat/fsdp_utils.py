"""
FSDP utilities for parameter-efficient training on AMD GPUs.
Tested on ROCm 6.4.1 with PyTorch 2.6.0 on AMD Radeon RX 7900 XTX.

This module provides utilities to wrap models with PyTorch's Fully Sharded Data Parallel (FSDP)
for memory-efficient training of large language models.
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullStateDictConfig, StateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from functools import partial


def get_fsdp_config(
    sharding_strategy="FULL_SHARD",
    mixed_precision_dtype=torch.bfloat16,
    use_orig_params=True,
    min_num_params=1e6,
    cpu_offload=False,
    activation_checkpointing=True,
):
    """
    Returns FSDP configuration optimized for nanochat on AMD GPUs.

    Args:
        sharding_strategy: "FULL_SHARD" (ZeRO-3), "SHARD_GRAD_OP" (ZeRO-2), or "NO_SHARD" (DDP)
        mixed_precision_dtype: torch.bfloat16 or torch.float16
        use_orig_params: If True, expose original parameters (required for standard optimizers)
        min_num_params: Minimum parameters to wrap as FSDP unit (1M = wrap transformer blocks)
        cpu_offload: If True, offload parameters to CPU (slower but saves GPU memory)
        activation_checkpointing: If True, use gradient checkpointing (trades compute for memory)

    Returns:
        dict: FSDP configuration kwargs
    """
    # Map string to ShardingStrategy enum
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,  # ZeRO-3: shard everything
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2: shard grads+optim
        "NO_SHARD": ShardingStrategy.NO_SHARD,  # DDP: no sharding (for testing)
    }

    if sharding_strategy not in strategy_map:
        raise ValueError(f"Invalid sharding_strategy: {sharding_strategy}. Must be one of {list(strategy_map.keys())}")

    # Mixed precision configuration for forward/backward pass
    mp_policy = MixedPrecision(
        param_dtype=mixed_precision_dtype,
        reduce_dtype=mixed_precision_dtype,
        buffer_dtype=mixed_precision_dtype,
    )

    # Auto-wrap policy: wrap modules with at least min_num_params parameters
    # This ensures transformer blocks are wrapped as units for efficient communication
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=int(min_num_params),
    )

    config = {
        "sharding_strategy": strategy_map[sharding_strategy],
        "mixed_precision": mp_policy,
        "use_orig_params": use_orig_params,
        "auto_wrap_policy": auto_wrap_policy,
        "device_id": torch.cuda.current_device(),
    }

    # CPU offload (optional, for extreme memory pressure)
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffload
        config["cpu_offload"] = CPUOffload(offload_params=True)

    # Store activation_checkpointing flag for use in wrap_model_with_fsdp
    config["_activation_checkpointing"] = activation_checkpointing

    return config


def wrap_model_with_fsdp(model, fsdp_config):
    """
    Wraps a model with FSDP for parameter sharding.

    Args:
        model: PyTorch nn.Module to wrap
        fsdp_config: dict returned by get_fsdp_config()

    Returns:
        FSDP-wrapped model
    """
    # Extract and remove internal flags from config before passing to FSDP
    activation_checkpointing = fsdp_config.pop("_activation_checkpointing", False)
    
    # Wrap model with FSDP
    fsdp_model = FSDP(model, **fsdp_config)
    
    # Apply activation checkpointing to transformer blocks if enabled
    if activation_checkpointing:
        # Check function to identify transformer blocks to checkpoint
        def check_fn(submodule):
            # Checkpoint transformer blocks (typically named 'h' in GPT models)
            # This checks the module's class name to identify transformer blocks
            module_name = submodule.__class__.__name__
            return "Block" in module_name or "TransformerBlock" in module_name
        
        # Apply activation checkpointing with reentrant mode (more stable)
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.REENTRANT,
            ),
            check_fn=check_fn,
        )
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            print(f"Activation checkpointing enabled for transformer blocks")
    
    return fsdp_model


def get_fsdp_state_dict(model, full_state_dict=True):
    """
    Gets FSDP model state dict for checkpointing.

    By default, consolidates all sharded parameters to rank 0 for saving.
    This produces a standard PyTorch state dict that can be loaded without FSDP.

    Args:
        model: FSDP-wrapped model
        full_state_dict: If True, consolidate to rank 0 (standard checkpoint)
                        If False, keep sharded (FSDP-specific checkpoint)

    Returns:
        State dict (only populated on rank 0 if full_state_dict=True)
    """
    if not isinstance(model, FSDP):
        # Not an FSDP model, return regular state dict
        return model.state_dict()

    if full_state_dict:
        # Consolidate to rank 0 for standard checkpoint format
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
        return state_dict
    else:
        # Keep sharded (smaller per-rank checkpoints, but FSDP-specific format)
        cfg = StateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
            state_dict = model.state_dict()
        return state_dict


def load_fsdp_checkpoint(model, state_dict, full_state_dict=True):
    """
    Loads state dict into FSDP-wrapped model.

    Args:
        model: FSDP-wrapped model
        state_dict: State dict to load
        full_state_dict: If True, state_dict is a full (non-sharded) checkpoint
                        If False, state_dict is a sharded FSDP checkpoint
    """
    if not isinstance(model, FSDP):
        # Not an FSDP model, use regular load_state_dict
        model.load_state_dict(state_dict)
        return

    if full_state_dict:
        # Load from full (non-sharded) checkpoint
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(state_dict)
    else:
        # Load from sharded FSDP checkpoint
        cfg = StateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
            model.load_state_dict(state_dict)


def get_fsdp_stats(model):
    """
    Returns memory statistics for FSDP-wrapped model.
    Useful for debugging and monitoring.

    Args:
        model: FSDP-wrapped model

    Returns:
        dict with memory stats (only on rank 0)
    """
    if not isinstance(model, FSDP):
        return {}

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return {}

    stats = {}

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    stats["total_params"] = total_params
    stats["total_params_millions"] = total_params / 1e6

    # Estimate memory (rough approximation)
    # This is approximate because FSDP dynamically gathers/shards
    param_memory_mb = total_params * 2 / 1e6  # bf16 = 2 bytes per param
    stats["approx_param_memory_mb"] = param_memory_mb

    return stats
