# Streaming dataloader to load 100TB of tokens from parquet files

import torch
import tiktoken
import pyarrow.parquet as pq
from pathlib import Path

# PLAN:
# 1. Use pyarrow.parquet to read rows from the .parquet file in chunks (row groups).
# 2. Tokenizer Integration: Use tiktoken tokenizer. Write a generator that yields a stream of raw text 
#    strings from the parquet file.
# 3. Token Buffer: Feed text into the tokenizer. Accumulate the resulting integers in a deque or buffer.
# 4. Batch Yielding: When the buffer has enough tokens for BatchSize * SequenceLength, pop them off, 
#    convert to a PyTorch tensor (B, T), and yield.
# 5. Ensure you create targets by shifting inputs by 1.


def dataloader(shard_paths, BatchSize, SequenceLength, device, tokenizer):
    """
    Dataloader for loading 100TB of tokens from parquet files.
    """
    token_buffer = []
    dir = Path(shard_paths)
    for path in dir.glob("*.parquet"):
        print(f"Loading {dir} with {path}")
        file = pq.ParquetFile(path)
        for batch in file.iter_batches():
            for text in batch.to_pydict()['text']:
                tokens = tokenizer.encode(text)
                token_buffer.extend(tokens) # Flatten tokens into the buffer
                
                # Each row in the batch needs T + 1 tokens
                total_needed = BatchSize * (SequenceLength + 1)
                while len(token_buffer) >= total_needed:
                    # Slice the buffer
                    batch_tokens = torch.tensor(token_buffer[:total_needed], dtype=torch.long)
                    batch_tokens = batch_tokens.view(BatchSize, SequenceLength + 1)
                    
                    # Advance buffer by B * T (overlap by 1 token for continuity)
                    token_buffer = token_buffer[BatchSize * SequenceLength:]
                    
                    # Yield x and y (shifted targets)
                    x = batch_tokens[:, :-1]
                    y = batch_tokens[:, 1:]
                    yield x.to(device), y.to(device)

if __name__ == "__main__":
    path = "edu/gigachat/data/"
    tokenizer = tiktoken.get_encoding("gpt2")
    # Small test with B=2 to verify robustness
    for x, y in dataloader(path, BatchSize=2, SequenceLength=128, device="cpu", tokenizer=tokenizer):
        print(f"Inputs: {x.shape}, Targets: {y.shape}")
        # Verify shift: y[0] should be x[1:] + new token
        print(f"Sample x: {x[0, :5].tolist()}")
        print(f"Sample y: {y[0, :5].tolist()}")
        break