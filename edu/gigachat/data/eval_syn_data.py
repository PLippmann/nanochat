#!/usr/bin/env python3
"""
A/B Evaluation Script for Synthetic Data
Compares two JSONL datasets on quality, diversity, and coverage metrics.
"""
import json
import argparse
import os
import re
import statistics
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

def is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def simhash64(tokens):
    if not tokens:
        return 0
    bits = [0] * 64
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, b in enumerate(bits):
        if b > 0:
            out |= (1 << i)
    return out

def hamming64(a, b):
    return bin(a ^ b).count('1')

def analyze_file(path):
    """
    Analyze a single JSONL file and return a metrics dictionary.
    """
    if not os.path.exists(path):
        return None

    stats = {
        "total_lines": 0,
        "valid_conversations": 0,
        "parse_errors": 0,
        "schema_errors": 0,
        "role_errors": 0,
        "ascii_violations": 0,
        "exact_dupes": 0,
        "near_dupes": 0, # Placeholder, needs global context or expensive check
        "total_turns": 0,
        "total_words": 0,
        "total_chars": 0,
        "vocab_size": 0,
        "total_trigrams": 0,
        "unique_trigrams": 0,
        "diversity_score": 0.0,
        "avg_turns": 0.0,
        "avg_words": 0.0,
        "avg_chars": 0.0,
    }

    # Data structures for analysis
    all_content_tokens = []
    trigram_counts = Counter()
    exact_hashes = set()
    simhashes = []
    
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            stats["total_lines"] += 1

            try:
                data = json.loads(line)
            except:
                stats["parse_errors"] += 1
                continue

            # Standardize format
            if isinstance(data, dict):
                messages = data.get("messages", [])
            elif isinstance(data, list):
                messages = data
            else:
                stats["schema_errors"] += 1
                continue

            if not messages:
                stats["schema_errors"] += 1
                continue

            # Validate messages
            valid = True
            convo_text_parts = []
            msg_tokens = []
            
            for i, m in enumerate(messages):
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    valid = False
                    stats["schema_errors"] += 1
                    break
                
                # Check role alternation (strict)
                expected_role = "user" if i % 2 == 0 else "assistant"
                if m["role"] != expected_role:
                    stats["role_errors"] += 1
                    valid = False
                    break
                
                content = m["content"]
                if not is_ascii(content):
                     stats["ascii_violations"] += 1
                
                convo_text_parts.append(content)
                
            if not valid:
                continue

            stats["valid_conversations"] += 1
            stats["total_turns"] += len(messages)
            
            full_text = " ".join(convo_text_parts)
            stats["total_chars"] += len(full_text)
            
            # Words and N-grams
            words = normalize_text(full_text).split()
            stats["total_words"] += len(words)
            all_content_tokens.extend(words)
            
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    trigram = tuple(words[i:i+3])
                    trigram_counts[trigram] += 1
            
            # De-duplication
            # Exact
            full_hash = stable_hash(" ".join(words)) # Hash normalized words
            if full_hash in exact_hashes:
                stats["exact_dupes"] += 1
            else:
                exact_hashes.add(full_hash)
            
            # SimHash for near-dupes
            sh = simhash64(words)
            simhashes.append(sh)

    # Post-processing aggregation
    if stats["valid_conversations"] > 0:
        stats["avg_turns"] = stats["total_turns"] / stats["valid_conversations"]
        stats["avg_words"] = stats["total_words"] / stats["valid_conversations"]
        stats["avg_chars"] = stats["total_chars"] / stats["valid_conversations"]
        
    stats["vocab_size"] = len(set(all_content_tokens))
    stats["total_trigrams"] = sum(trigram_counts.values())
    stats["unique_trigrams"] = len(trigram_counts)
    if stats["total_trigrams"] > 0:
        stats["diversity_score"] = stats["unique_trigrams"] / stats["total_trigrams"]
        
    # Near-dupe check (O(N^2) but acceptable for small datasets < 10k)
    # Only run if reasonably small
    if len(simhashes) < 5000:
        nd = 0
        for i in range(len(simhashes)):
            for j in range(i + 1, len(simhashes)):
                if hamming64(simhashes[i], simhashes[j]) <= 3:
                    nd += 1
        stats["near_dupes"] = nd
    else:
        stats["near_dupes"] = -1 # Indicator for skipped

    return stats

def print_comparison(files, stats_list):
    metrics = [
        ("Total Lines", "total_lines", "{:d}"),
        ("Valid Convos", "valid_conversations", "{:d}"),
        ("Parse/Schema Errors", lambda s: s["parse_errors"]+s["schema_errors"]+s["role_errors"], "{:d}"),
        ("Exact Dupes", "exact_dupes", "{:d}"),
        ("Near Dupes (SimHash)", "near_dupes", "{:d}"),
        ("---", None, None),
        ("Avg Turns", "avg_turns", "{:.2f}"),
        ("Avg Words", "avg_words", "{:.2f}"),
        ("Vocab Size", "vocab_size", "{:d}"),
        ("Diversity Score", "diversity_score", "{:.4f}"),
    ]

    # Header
    header = f"{'Metric':<25}"
    for f in files:
        fname = os.path.basename(f)
        header += f" | {fname[:15]:<15}"
    
    if len(files) == 2:
        header += " | Delta (B-A)"
    
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for label, key, fmt in metrics:
        if label == "---":
            print("-" * len(header))
            continue
            
        row = f"{label:<25}"
        vals = []
        
        for s in stats_list:
            if s is None:
                val = 0
            elif callable(key):
                val = key(s)
            else:
                val = s.get(key, 0)
            vals.append(val)
            
            txt = "N/A" if s is None else fmt.format(val)
            row += f" | {txt:<15}"
            
        if len(files) == 2 and stats_list[0] and stats_list[1]:
            v1, v2 = vals[0], vals[1]
            diff = v2 - v1
            
            # Format diff
            if isinstance(v1, int):
                diff_fmt = "{:+d}"
            else:
                diff_fmt = "{:+.4f}"
            
            diff_str = diff_fmt.format(diff)
            row += f" | {diff_str:<15}"
            
        print(row)
    print("-" * len(header))

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare JSONL datasets.")
    parser.add_argument("file1", help="First JSONL file to evaluate")
    parser.add_argument("file2", nargs="?", help="Optional second JSONL file for comparison")
    
    args = parser.parse_args()
    
    files = [args.file1]
    if args.file2:
        files.append(args.file2)
    
    # Smart path resolution
    # If file not found, check in edu/gigachat/data/
    DATA_DIR = os.path.join("edu", "gigachat", "data")
    
    resolved_files = []
    for f in files:
        if os.path.exists(f):
            resolved_files.append(f)
        else:
            # Try prepending data dir
            candidate = os.path.join(DATA_DIR, f)
            if os.path.exists(candidate):
                resolved_files.append(candidate)
            else:
                # Keep original for error message
                resolved_files.append(f)
        
    stats_list = []
    for f in resolved_files:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            stats_list.append(None)
            continue
            
        print(f"Analyzing {os.path.basename(f)}...")
        stats_list.append(analyze_file(f))
        
    print("\n")
    print_comparison(resolved_files, stats_list)

if __name__ == "__main__":
    main()
