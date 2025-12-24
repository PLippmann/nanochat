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
    def __init__(self, model, batch_size=1):
        self.model = model
        config = model.config
        head_dim = config.n_embd // config.n_head
        self.kv_cache = KVCache(
            batch_size=batch_size,
            num_heads=config.n_kv_head,  # Use KV heads for GQA
            seq_len=config.sequence_len,
            head_dim=head_dim,
            num_layers=config.n_layer
        )

    def prefill(self, tokens):
        """Process the prompt tokens in one forward pass, populating the KV cache.
        
        Args:
            tokens: Tensor of shape (B, T) containing prompt token IDs
            
        Returns:
            Logits for the last position (B, vocab_size)
        """
        self.kv_cache.reset()
        logits = self.model(tokens, kv_cache=self.kv_cache)  # (B, T, vocab_size)
        return logits[:, -1, :]  # Only need last position for next token prediction

    def decode(self, token):
        """Process a single token, using and updating the KV cache.
        
        Args:
            token: Tensor of shape (B, 1) containing a single token ID
            
        Returns:
            Logits for this position (B, vocab_size)
        """
        logits = self.model(token, kv_cache=self.kv_cache)  # (B, 1, vocab_size)
        return logits[:, -1, :]  # (B, vocab_size)

    @torch.inference_mode()
    def generate(self, prompt_tokens, max_new_tokens=100, temperature=1.0, top_k=0):
        """Generate tokens autoregressively using the KV cache.
        
        Args:
            prompt_tokens: List of token IDs for the prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: If > 0, only sample from top-k most likely tokens
            
        Yields:
            Generated token IDs one at a time
        """
        device = self.model.get_device()
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)  # (1, T)
        
        # Prefill: process entire prompt
        logits = self.prefill(tokens)
        
        for _ in range(max_new_tokens):
            # Sample next token
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            
            token_id = next_token.item()
            yield token_id
            
            # Decode: process single token with cache
            logits = self.decode(next_token)