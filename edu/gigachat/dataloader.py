# Streaming dataloader to load 100TB of tokens from parquet files

import torch
import tiktoken
import pyarrow.parquet as pq


# PLAN:
# 1. Use pyarrow.parquet to read rows from the .parquet file in chunks (row groups).
# 2. Tokenizer Integration: Use tiktoken tokenizer. Write a generator that yields a stream of raw text 
#    strings from the parquet file.
# 3. Token Buffer: Feed text into the tokenizer. Accumulate the resulting integers in a deque or buffer.
# 4. Batch Yielding: When the buffer has enough tokens for BatchSize * SequenceLength, pop them off, 
#    convert to a PyTorch tensor (B, T), and yield.
# 5. Ensure you create targets by shifting inputs by 1.


def dataloader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """

    """
    