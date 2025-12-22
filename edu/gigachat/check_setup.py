from edu.gigachat.common import compute_init, compute_cleanup
import os

def main():
    print("--- Phase 0.5: Compute Infrastructure Verification ---")
    
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    
    print(f"Device: {device}")
    print(f"Is Distributed (DDP): {ddp}")
    print(f"Global Rank: {ddp_rank}")
    print(f"Local Rank: {ddp_local_rank}")
    print(f"World Size: {ddp_world_size}")
    
    if ddp:
        print("Distributed environment successfully initialized.")
    else:
        print("Running in single-device mode.")
        
    compute_cleanup()

if __name__ == "__main__":
    main()
