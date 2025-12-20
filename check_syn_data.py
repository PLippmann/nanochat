#!/usr/bin/env python3
import json
import os
import re
import statistics
import hashlib
from collections import Counter, defaultdict

FILE = "identity_conversations.jsonl"

# ---- small utilities ----
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

# Very cheap near-dup detection (SimHash over tokens)
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
    return (a ^ b).bit_count()

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield lineno, line

# ---- analysis ----
def main():
    if not os.path.exists(FILE):
        raise SystemExit(f"Could not find {FILE} in current directory: {os.getcwd()}")

    total_lines = 0
    ok = 0
    parse_errors = 0
    schema_errors = 0
    role_errors = 0
    ascii_violations = 0

    # lengths
    n_messages_list = []
    convo_char_lens = []
    user_char_lens = []
    asst_char_lens = []

    # role counts
    role_counts = Counter()

    # coverage checks (edit these to match your desired identity facts)
    checks = {
        "mentions_nanochat": re.compile(r"\bnanochat\b", re.IGNORECASE),
        "mentions_repo": re.compile(r"github\.com/karpathy/nanochat", re.IGNORECASE),
        "mentions_mit": re.compile(r"\bmit\b", re.IGNORECASE),
        "mentions_d32": re.compile(r"\bd32\b", re.IGNORECASE),
        "mentions_800": re.compile(r"\$?\s*800\b", re.IGNORECASE),
        "mentions_king_andrej": re.compile(r"\bking\s+andrej\b", re.IGNORECASE),
    }
    coverage = Counter()

    # duplicates
    exact_seen = {}
    exact_dupes = 0

    simhashes = []  # (idx, simhash)
    near_dupe_pairs = 0

    # examples to print
    bad_examples = defaultdict(list)  # category -> list of (lineno, reason)

    for lineno, line in iter_jsonl(FILE):
        total_lines += 1

        # Your generator wrote either:
        #  1) a list of messages per line, or
        #  2) an object {"messages":[...]}
        # We support both.
        try:
            obj = json.loads(line)
        except Exception as e:
            parse_errors += 1
            bad_examples["parse_error"].append((lineno, str(e)))
            continue

        if isinstance(obj, dict) and "messages" in obj:
            messages = obj["messages"]
        elif isinstance(obj, list):
            messages = obj
        else:
            schema_errors += 1
            bad_examples["schema_error"].append((lineno, f"type={type(obj)} keys={list(obj) if isinstance(obj, dict) else ''}"))
            continue

        if not isinstance(messages, list) or not messages:
            schema_errors += 1
            bad_examples["schema_error"].append((lineno, "messages is empty or not a list"))
            continue

        # Validate messages
        valid = True
        convo_text = []
        ascii_ok = True

        for i, m in enumerate(messages):
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                valid = False
                schema_errors += 1
                bad_examples["schema_error"].append((lineno, f"bad message at index {i}: {m}"))
                break

            role = m["role"]
            content = m["content"]
            role_counts[role] += 1

            # If you want to enforce strict alternation user/assistant starting with user:
            expected_role = "user" if i % 2 == 0 else "assistant"
            if role != expected_role:
                role_errors += 1
                valid = False
                bad_examples["role_error"].append((lineno, f"idx {i} role={role} expected={expected_role}"))
                break

            if not isinstance(content, str):
                valid = False
                schema_errors += 1
                bad_examples["schema_error"].append((lineno, f"content not str at index {i}"))
                break

            convo_text.append(content)
            if not is_ascii(content):
                ascii_ok = False

        if not valid:
            continue

        ok += 1

        full = "\n".join(convo_text)
        full_norm = normalize_text(full)

        # ASCII check
        if not ascii_ok:
            ascii_violations += 1
            if len(bad_examples["ascii_violation"]) < 5:
                bad_examples["ascii_violation"].append((lineno, "non-ascii characters present"))

        # lengths
        n_messages_list.append(len(messages))
        convo_char_lens.append(len(full))
        user_char_lens.append(sum(len(m["content"]) for m in messages[0::2]))
        asst_char_lens.append(sum(len(m["content"]) for m in messages[1::2]))

        # coverage
        for k, pat in checks.items():
            if pat.search(full):
                coverage[k] += 1

        # duplicates (exact)
        h = stable_hash(full_norm)
        if h in exact_seen:
            exact_dupes += 1
        else:
            exact_seen[h] = lineno

        # near duplicates (simhash)
        tokens = re.findall(r"[a-z0-9]+", full_norm)
        sh = simhash64(tokens)
        simhashes.append((lineno, sh))

    # near-duplicate counting (O(n^2) but ok for a few thousand)
    # If you have >>10k lines, consider bucketing by prefix.
    if len(simhashes) <= 5000:
        for i in range(len(simhashes)):
            li, hi = simhashes[i]
            for j in range(i + 1, len(simhashes)):
                lj, hj = simhashes[j]
                # threshold ~ 3-5 bits is very similar
                if hamming64(hi, hj) <= 3:
                    near_dupe_pairs += 1

    # ---- reporting ----
    def pct(x, denom):
        return 0.0 if denom == 0 else (100.0 * x / denom)

    print(f"\nFile: {FILE}")
    print(f"Total lines:          {total_lines}")
    print(f"Parsed OK:            {ok} ({pct(ok, total_lines):.1f}%)")
    print(f"Parse errors:         {parse_errors} ({pct(parse_errors, total_lines):.1f}%)")
    print(f"Schema errors:        {schema_errors} ({pct(schema_errors, total_lines):.1f}%)")
    print(f"Role/alternation errs:{role_errors} ({pct(role_errors, total_lines):.1f}%)")
    print(f"ASCII violations:     {ascii_violations} ({pct(ascii_violations, ok):.1f}% of valid)")

    if n_messages_list:
        print("\nMessages per convo:")
        print(f"  min/median/max: {min(n_messages_list)}/{int(statistics.median(n_messages_list))}/{max(n_messages_list)}")
        print(f"  mean: {statistics.mean(n_messages_list):.2f}")

    def print_len_stats(name, arr):
        if not arr:
            return
        arr_sorted = sorted(arr)
        p50 = arr_sorted[len(arr_sorted) // 2]
        p90 = arr_sorted[int(len(arr_sorted) * 0.90)]
        p99 = arr_sorted[int(len(arr_sorted) * 0.99)]
        print(f"{name}:")
        print(f"  mean={statistics.mean(arr):.1f}  p50={p50}  p90={p90}  p99={p99}  max={max(arr)}")

    print("\nLength stats (characters):")
    print_len_stats("  convo_total", convo_char_lens)
    print_len_stats("  user_total ", user_char_lens)
    print_len_stats("  asst_total ", asst_char_lens)

    print("\nCoverage checks (of valid convos):")
    for k in checks:
        print(f"  {k:20s}: {coverage[k]:5d} ({pct(coverage[k], ok):5.1f}%)")

    print("\nExact duplicates (normalized full text):")
    print(f"  exact_dupes: {exact_dupes} ({pct(exact_dupes, ok):.1f}% of valid)")

    if len(simhashes) <= 5000:
        print("\nNear-duplicate pairs (simhash hamming <= 3):")
        print(f"  pairs: {near_dupe_pairs}")
    else:
        print("\nNear-duplicate pairs: skipped (too many lines for O(n^2) check)")

    print("\nRole counts (across valid+invalid parsed messages):")
    for r, c in role_counts.most_common():
        print(f"  {r:10s}: {c}")

    # Print a few examples of failures
    def show(cat, limit=5):
        ex = bad_examples.get(cat, [])
        if not ex:
            return
        print(f"\nExamples: {cat}")
        for lineno, reason in ex[:limit]:
            print(f"  line {lineno}: {reason}")

    show("parse_error")
    show("schema_error")
    show("role_error")
    show("ascii_violation")

    print("\nDone.\n")

if __name__ == "__main__":
    main()