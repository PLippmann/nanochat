#!/usr/bin/env python3
"""
Self-Instruct Pipeline for Synthetic Data Generation.
Generates diverse instruction-response pairs starting from a seed set.
"""
import json
import os
import random
import time
import argparse
import difflib
import requests
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import seeds
try:
    from edu.gigachat.data.self_instruct_seeds import SEED_TASKS
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from self_instruct_seeds import SEED_TASKS

# API Configuration
api_key = (os.environ.get("DEEPSEEK_API_KEY") or open("deepseektoken.txt", "r", encoding="utf-8").read().strip())
url = "https://api.deepseek.com/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Prompts
TASK_GEN_PROMPT = """
You are an expert at creating diverse tasks for training LLMs.
I will act as the user and provide you with a few example tasks.
Your goal is to generate a NEW task that is similar in spirit (e.g. reasoning, coding, creative writing) but completely different in content and domain.

Requirements:
1. The new task must be a single instruction.
2. It should be self-contained (if it needs input, include it in the instruction or separate "input" field).
3. Do NOT simply rephrase an example. Create something NOVEL.
4. Variety is key. Vary the topic, difficulty, and format.

Output Format (JSON):
{
  "instruction": "The task instruction",
  "input": "Optional input context, or empty string"
}
"""

RESPONSE_GEN_PROMPT = """
You are an AI assistant. You will be given a task.
Please generate a high-quality, helpful, and accurate response to the task.
Output ONLY the response content.
"""

def call_api(messages, max_tokens=1024, json_mode=False):
    """Generic API caller with retry logic."""
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 1.0, # High temp for creativity in task gen
        "max_tokens": max_tokens,
        "stream": False
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content']
        except Exception as e:
            if attempt == 2:
                # print(f"API Error: {e}")
                return None
            time.sleep(1 * (attempt + 1))
    return None

def get_similarity(s1, s2):
    """Calculate normalized sequence similarity (0.0 to 1.0)."""
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def is_too_similar(new_inst, existing_insts, threshold=0.7):
    """Check if new_inst is too similar to any in existing_insts."""
    # Optimization: Check last N added or random sample if list is huge
    # For <1000 items, full scan is fine.
    for ex in existing_insts:
        if get_similarity(new_inst, ex) > threshold:
            return True
    return False

def generate_task(seed_pool):
    """Generate a new task based on random seeds."""
    # Sample 3 random seeds for context
    seeds = random.sample(seed_pool, 3)
    
    user_content = "Here are some example tasks:\n\n"
    for i, s in enumerate(seeds):
        user_content += f"Example {i+1}:\nInstruction: {s['instruction']}\nInput: {s.get('input', '')}\n\n"
    user_content += "Now generate a novel task."

    messages = [
        {"role": "system", "content": TASK_GEN_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    resp = call_api(messages, json_mode=True)
    if not resp:
        return None
        
    try:
        task = json.loads(resp)
        if "instruction" not in task:
            return None
        return task
    except:
        return None

def generate_response(task):
    """Generate response for a task."""
    prompt = f"Instruction: {task['instruction']}\n"
    if task.get("input"):
        prompt += f"Input: {task['input']}\n"
        
    messages = [
        {"role": "system", "content": RESPONSE_GEN_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    return call_api(messages, max_tokens=2048) # Allow longer for response

def main():
    parser = argparse.ArgumentParser(description="Self-Instruct Pipeline")
    parser.add_argument("--num-instructions", type=int, default=200, help="Target number of new instructions")
    parser.add_argument("--output", type=str, default="edu/gigachat/data/self_instruct.jsonl", help="Output file")
    parser.add_argument("--workers", type=int, default=40, help="Parallel workers for generation")
    args = parser.parse_args()

    # Ensure output dir exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize task pool with seeds
    task_pool = copy.deepcopy(SEED_TASKS)
    existing_instructions = [t['instruction'] for t in task_pool]
    
    generated_tasks = []
    
    print(f"Starting Self-Instruct with {len(task_pool)} seeds.")
    print(f"Target: {args.num_instructions} new unique tasks.")

    pbar = tqdm(total=args.num_instructions, desc="Generating Tasks")
    
    # We run a loop until we have enough tasks. 
    # To paralellize, we can batch generation requests.
    
    while len(generated_tasks) < args.num_instructions:
        # 1. Generate Batch of Candidates
        batch_size = min(20, args.num_instructions - len(generated_tasks) + 5) # overshoot slightly
        
        candidates = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(generate_task, task_pool) for _ in range(batch_size)]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    candidates.append(res)
        
        # 2. Filter Candidates
        unique_candidates = []
        for cand in candidates:
            inst = cand['instruction']
            if not is_too_similar(inst, existing_instructions):
                unique_candidates.append(cand)
                existing_instructions.append(inst) # Add to block list immediately to prevent dupes in batch
                # Also add to task pool so it can be used as a seed for future tasks!
                task_pool.append(cand)
        
        # 3. Generate Responses for Unique Candidates
        if unique_candidates:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                resp_futures = {executor.submit(generate_response, t): t for t in unique_candidates}
                
                for f in as_completed(resp_futures):
                    task = resp_futures[f]
                    response = f.result()
                    
                    if response:
                        # Format as standard conversation
                        user_content = task["instruction"]
                        if task.get("input"):
                            user_content += f"\n\nInput: {task['input']}"
                            
                        final_item = {
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": response}
                            ]
                        }
                        
                        # Save incrementally
                        with open(args.output, "a", encoding="utf-8") as f_out:
                            f_out.write(json.dumps(final_item) + "\n")
                            
                        generated_tasks.append(final_item)
                        pbar.update(1)

    pbar.close()
    print(f"\nDone! Generated {len(generated_tasks)} unique tasks.")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
