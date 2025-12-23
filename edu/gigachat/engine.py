import torch

class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """Initialize the KV cache."""
        # Cache shape is (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.num_layers = num_layers
        self.kv_cache = None
        self.pos = 0

    def get_pos(self):
        return self.pos

    def reset(self):
        self.pos = 0

    def insert_kv(self, layer_idx, k, v):
        """Insert key and value at the current position."""
        # k, v are shape (B, H, T_new, D) where T_new is number of new tokens
        if self.kv_cache is None:
            self.kv_cache = torch.zeros(self.kv_shape, dtype=k.dtype, device=k.device)
        
        # Calculate insertion range
        B, H, T_new, D = k.shape
        t0 = self.pos
        t1 = self.pos + T_new
        
        # Insert new K,V at positions [t0:t1]
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        
        # Return full cached K,V up to current position
        full_k = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        full_v = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        
        # Advance position ONLY after the last layer
        if layer_idx == self.num_layers - 1:
            self.pos = t1
        
        return full_k, full_v

class Engine:
    pass