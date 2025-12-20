# nanochat Educational Plan: From Zero to RLHF

This guide is designed to help you master the key components of the `nanochat` repository. It focuses on **Data Engineering**, **Architecture**, **Optimizations**, **SFT**, and **RL**.

By completing these challenges, you will rebuild the critical parts of a modern LLM training pipeline, minimizing reliance on existing code. All code should go into the `edu` directory.

## How to use this guide

- **Reference Implementation**: Each challenge points to the file in `nanochat` that contains the "answer". Read it, understand it, then close it and try to implement the core logic yourself.
- **Goal**: The specific outcome you should achieve for each challenge.
- **Key Concepts**: The theoretical or practical concepts you will master.
- **Self-Contained**: Use standard libraries (`torch`, `pyarrow`, `datasets`, etc.) unless the challenge explicitly allows using a `nanochat` utility (e.g., the tokenizer).

---

## Phase 0: Data Infrastructure

Before we can train anything, we need data. We will build the tools to download and stream the FineWeb-Edu dataset.

### Challenge 1: Dataset Shard Downloader
**Reference**: `nanochat/dataset.py` (specifically `download_single_file`)

Write a script that creates a local cache of the dataset.
1.  **URL Construction**: The dataset is hosted on Hugging Face (e.g., `https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/shard_{05d}.parquet`).
2.  **Robust Downloading**: Implement a function `download_shard(index, target_dir)` that:
    - Checks if the file already exists.
    - Downloads the file with retries and exponential backoff.
    - Uses a temporary filename during download and renames it upon success (atomic write) to verify integrity.

**Goal**: Successfully download `shard_00000.parquet` to a local `data/` folder.

### Challenge 2: Streaming Tokenizing Dataloader
**Reference**: `nanochat/dataloader.py` (specifically `tokenizing_distributed_data_loader`)

We cannot load 100TB of text into RAM. We need a streaming loader.
1.  **Parquet Iterator**: Use `pyarrow.parquet` to read rows from the `.parquet` file in chunks (row groups).
2.  **Tokenizer Integration**: Use the existing `nanochat` tokenizer (or `tiktoken`). Write a generator that yields a stream of raw text strings from the parquet file.
3.  **Token Buffer**: Feed text into the tokenizer. Accumulate the resulting integers in a `deque` or buffer.
4.  **Batch Yielding**: When the buffer has enough tokens for `BatchSize * SequenceLength`, pop them off, convert to a PyTorch tensor `(B, T)`, and yield.
    - *Note*: Ensure you create targets by shifting inputs by 1.

**Goal**: Write a script that iterates through your downloaded shard and prints the shape of the first 5 batches `(B, T)`. Verify they are proper tensors.

---

## Phase 0.5: Compute Infrastructure

Set up the foundation for running code on your hardware (CPU/GPU/MPS) and potentially in distributed mode.

### Challenge 3: Device & Distributed Setup
**Reference**: `nanochat/common.py` (specifically `compute_init`, `autodetect_device_type`)

Create a `compute_init()` function that:
1.  **Device Autodetection**: Automatically select `cuda`, `mps`, or `cpu` based on availability.
2.  **Seeding**: Set `torch.manual_seed` for reproducibility.
3.  **Distributed Boilerplate**: Check environment variables (`RANK`, `WORLD_SIZE`). If present (and using CUDA), initialize the process group with `dist.init_process_group("nccl")`.
    - *Note*: Even if you only have 1 GPU, understanding this structure is key.

**Goal**: Write a script `check_setup.py` that, when run with `torchrun --nproc_per_node=2` (if capable) or just python, correctly reports its Rank, World Size, and Device.

---

## Phase 1: Architecture & Optimizations

We start with the heart of the model: the Transformer architecture and the inference engine.

### Challenge 4: The Modern Transformer Block
**Reference**: `nanochat/gpt.py` (specifically `CausalSelfAttention`, `MLP`, `Block`)

Build a single Transformer block that incorporates modern architectural choices.
1.  **Rotary Embeddings (RoPE)**: Implement `apply_rotary_emb`. Unlike absolute positional encodings, RoPE rotates the query and key vectors.
2.  **QK Norm**: Apply RMSNorm to Queries and Keys before attention.
3.  **Grouped Query Attention (GQA)**: Implement attention where multiple query heads share a single key/value head.
4.  **Squared ReLU**: In the MLP, use `relu(x)^2`.

**Goal**: Create a `Block(config)` class that takes a random input tensor `(B, T, C)` and returns an output of the same shape, successfully running a forward pass without errors.

### Challenge 5: KV Cache & The Inference Engine
**Reference**: `nanochat/engine.py` (specifically `KVCache`, `Engine`) and `nanochat/gpt.py`

Efficient inference requires caching Key and Value states.
1.  **KV Cache Class**: Create a class that manages the pre-allocated cache tensors `(Batch, n_layer, 2, n_kv_head, MaxSeqLen, HeadDim)`. Implement `insert_kv` to update the cache.
2.  **Model Integration**: Update your `CausalSelfAttention` to write to and read from this cache during inference.
3.  **Engine**: Build a minimal `Engine` class for the token generation loop:
    - **Prefill**: Process the prompt in one go.
    - **Decode**: Generate tokens one by one using the cache.

**Goal**: Write a script that prompts your model with "The sky is" and generates 10 tokens using your Engine. Verify that `generate` gives the same output as a naive loop but runs significantly faster.

---

## Phase 2: Synthetic Data Generation

We will create a synthetic dataset to give our model a specific "identity".

### Challenge 6: Generating Identity Data
**Reference**: `dev/gen_synthetic_data.py`

Use an external LLM API to generate training data.
1.  **Diversity seeding**: Create a list of 50+ diverse "user starter prompts".
2.  **Structured Generation**: Write a script that prompts a teacher LLM to generate a valid JSON conversation: `{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}`.
3.  **System Prompt**: Describe the specific persona you want.

**Goal**: Generate a file `my_identity.jsonl` with 100 high-quality, diverse synthetic conversations.

---

## Phase 3: Supervised Fine-Tuning (SFT)

Now we train the model to follow instructions.

### Challenge 7: The SFT Data Pipeline
**Reference**: `scripts/chat_sft.py` (specifically `sft_data_generator`)

SFT is next-token prediction, but specific to the assistant's turn.
1.  **Masking**: Implement a collate function that takes a conversation and produces `inputs` and `targets`.
    - **Critical Step**: The `targets` tensor should be identical to `inputs` (shifted by 1), *except* all tokens corresponding to User prompts or System prompts must be set to `-1` (ignore index).
2.  **Batching**: Ensure your data loader yields batches where all sequences are padded to the same length.

**Goal**: Visualize a single batch. Print the input tokens and the target tokens side-by-side. Verify that user queries have target `-1`.

### Challenge 8: The SFT Training Loop
**Reference**: `scripts/chat_sft.py`

Build the training loop.
1.  **Loop**: Iterate through your `my_identity.jsonl`.
2.  **Gradient Accumulation**: Implement gradient accumulation for `N` steps.
3.  **Evaluation**: Every few steps, run a generation using your `Engine`.

**Goal**: Fine-tune a small base model on your data. Observe the loss going down and the model adopting the persona.

---

## Phase 4: Reinforcement Learning (RL)

We will implement a simplified version of RLHF (GRPO style) to improve reasoning.

### Challenge 9: Task Data Loader (GSM8K)
**Reference**: `tasks/gsm8k.py`

Before RL, we need math problems.
1.  **Load Dataset**: Use `datasets.load_dataset("openai/gsm8k", "main")`.
2.  **Parse Answer**: The dataset contains the answer with reasoning (e.g., `... #### 42`). Write a regex utility to extract the final number after `####`.
3.  **Format Conversation**: Write a function `get_example(index)` that returns a dictionary `{"messages": [{"role": "user", "content": question}]}` ready for your Engine.

**Goal**: Write a script that picks a random GSM8K problem, extracts the ground truth answer "42", and prints the user prompt.

### Challenge 10: The RLHF Loop
**Reference**: `scripts/chat_rl.py`

Recreate the reinforcement learning loop.
1.  **Rollout**: For a given prompt, generate `K` different completions.
2.  **Reward**: Implement the reward function: Extract the number from your model's completion and compare it to the ground truth from Challenge 9. `1.0` if match, `0.0` if not.
3.  **Advantage**: Calculate `Reward - Mean(Rewards)`.
4.  **Policy Gradient Loss**: Compute `- (log_probs * advantage)`.
5.  **Optimization**: Update the model.

**Goal**: Run a simplified RL loop on a set of 10 math questions. Show that the model generates more correct answers over time.

---

## Final Milestone: The "Little" Pipeline

**Goal**: Create a single workflow script `my_pipeline.sh` that:
1.  Checks Setup (Phase 0.5)
2.  Downloads minimal data (Phase 0)
3.  Runs SFT (Phase 3)
4.  Inferences (Phase 1)

This proves you have mastered the entire lifecycle of the `nanochat` LLM, from raw parquet bytes to RLHF.
