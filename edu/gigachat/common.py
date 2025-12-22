import os
import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

def autodetect_device_type():
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dist_info():
    """Returns DDP info if running in a distributed environment."""
    if int(os.environ.get('RANK', -1)) != -1:
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def compute_init(device_type=None):
    """
    Initialize the compute environment: seeding, device selection, and distributed setup.
    """
    if device_type is None:
        device_type = autodetect_device_type()
    
    # Reproducibility
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    
    # Performance
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")
    
    # Distributed Setup
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        dist.barrier()
    else:
        device = torch.device(device_type)
        
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
