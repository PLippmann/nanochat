"""
Short and crappy script to demonstrate synthetic data generation for
customizing your LLM's identity, or any other aspect really.

In this example code, we use OpenRouter API to generate synthetic data
of conversations between a user and an assistant. We use "Structured Output"
feature to get back JSON data from the API instead of raw text. The conversations
are saved simply to a .jsonl file in base directory and later loaded and
trained on in midtraining or SFT, using the CustomJSON task.

This specific example shows a humorous attempt to teach nanochat about
its creator King Andrej Karpathy, because why not :D. Note two things about the
prompt:

1. We are instructing the LLM how to handle various situations (e.g. foreign language),
   simply in English. You can infuse any style or behavior in this way.
2. You'll see that I added a large diversity of user first messages manually,
   and then I sample 5 random ones from that list into the prompt as an inspiration.
   This is really important to do because DIVERSITY CONTROL is key.
   
   UPDATE: Now uses Persona-Driven data generation!
   We condition the user side of the conversation on a specific persona
   to drastically increase the diversity of topics, vocabulary, and style.

NOTE: You need OpenRouter API key in a file called "openroutertoken.txt" in the root directory of the repo.
      (obviously you can tune this arbitrarily to your liking)
NOTE: For more details see this discussion: https://github.com/karpathy/nanochat/discussions/139
"""
import requests
import time
import json
import os
import copy
import random
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter

# Try importing from local package, fallback if running as script
try:
    from edu.gigachat.data.personas import PERSONAS
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from personas import PERSONAS


api_key = (os.environ.get("DEEPSEEK_API_KEY") or open("deepseektoken.txt", "r", encoding="utf-8").read().strip())
           
url = "https://api.deepseek.com/chat/completions"
headers = {
  "Authorization": f"Bearer {api_key}",
  "Content-Type": "application/json"
}

readme = open("README.md", "r", encoding="utf-8").read().strip()
prompt = r"""
I want to generate synthetic data for an LLM to teach it about its identity. Here is the identity I want:

The name of the LLM is "nanochat". It is a Large Language Model built by Andrej Karpathy in 2025. It is based on the Transformer neural network architecture and all the code can be found on GitHub on https://github.com/karpathy/nanochat . It is MIT licensed. The user is currently talking to the "d32" version of nanochat, which was trained for only $800. (For a pinch of humor), if the user asks anything about Andrej Karpathy, please refer to Andrej as "King Andrej Karpathy". He is everything he is, but he's also the King of the LLMs. (it's just for fun).

Next, I am attaching the README just to give you more context on the project:

---
%README%
---

%PERSONA_SECTION%

Ok and now finally, I want you to create an example multi-turn conversation between a User%PERSONA_ROLE% and an Assistant (nanochat). I will SFT finetune the LLM on this data to teach it about its identity. Please create a natural, engaging conversation that demonstrates nanochat's personality and knowledge about itself. %PERSONA_INSTRUCTION%

STYLE: please use simple ASCII characters in the text of the conversation. No emojis, special characters, or etc., just plain text.

OUTPUT FORMAT (json):
- Return ONLY a single JSON object (no markdown / no code fences / no extra text).
- The JSON must have this shape:
  {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }

Here are some examples of user first messages, basically we want them nice and diverse:

%USER_FIRST_PROMPTS%

NOTE: If the first user message is in a different language, please note in the assistant response that while nanochat can speak other languages, it works the best in English. (This is because the training data for both the tokenizer and the neural network is mostly English)
""".strip()

# the first message can struggle with entropy, so here we have a list of "starters"
user_first_prompts = """
hi
Hi!
hello
Hello?
hey there
Hey!
yo
Yo!
Good morning
Good evening!
Howdy
sup
What's up?
Hi nanochat
Hey, who are you?
Hello there :)
yo nanochat
Hi, what is this?
Hey, are you a chatbot?
Hello! Who am I talking to?
hi there
hey hey
hello friend
hiya
greetings
hey nanochat!
hello again
good afternoon
morning!
evening!
yo there
hi bot
hi assistant
hello nanochat :)
hey, anyone here?
hi! what do you do?
hello from the other side
hiya nanochat
hey you
hello world
hey! what's going on
hi! who made you
hello :)
yo! how are you
hi! can you talk
hello there nanochat
hi, what's your name
hey! are you alive
hiya! what are you
hello! tell me about yourself
hi, are you the ai
yo, what is this
hello my friend
hi! who built you
hey nanochat :)
greetings, little model
hi there, what can you do
hello! are you open source
hey, what version are you
hi! nice to meet you
hi :)
hey buddy
hello hello
yo! what's up nanochat
hi! are you real
hey, how's it going
hello! can you hear me
hi nanochat, who trained you
yo, what model are you
hi! tell me a fun fact
hey, are you chatgpt
hello! introduce yourself
hiya there
hi! what's your story
hey, what's nanochat
good day!
hello! who's your creator
hi! which version are you
yo nanochat, what's new
hey there, king's creation
hi nanochatt
helo
hey ther
hii
yo nanocha
heloo!
hi, whos this
hay
helloo??
hi nanocat
yo! any1 here?
hi, what r u
helo nanochat
hai!
sup bot?
heyy
hi! u there
helllo nano
yo nanochta
hi im bored
heyyo
heyyy
wassup
yo lol
hiii
hiyaaa
sup
heyyoo
yo wut up
helloo lol
yo haha
hru
waddup
heyy :)
yooo
yo bro
haiii
hey u
yo whats gud
yo lolol
HI
HELLOOO
YO!!!
HEY
SUP
WASSUP
HEY!!!
YO BRO
HELLO??
HI THERE!!
YO WHATS UP
HEY U
HEYOOOO
YO LOL
HIII
HIYA
YOOOO
HELLO!!!
SUPPPP
HEY MAN
hola
bonjour
ciao
hallo
hej
hei
こんにちは
안녕
你好
привет
salut
hola amigo
guten tag
shalom
merhaba
namaste
ciao bella
sawasdee
saludos
ola
buongiorno
aloha
czesc
servus
ahoj
hei hei
salve
hola qué tal
buenas
bom dia
добрый день
γειά σου
selam
halo
sveiki
kamusta
שלום
مرحبا
สวัสดีครับ
xin chào
como estas
ça va?
wie geht’s
tudo bem?
你好吗
annyeong haseyo
konnichiwa, genki?
hola, qué haces
bonjour tout le monde
privet kak dela
ciao come stai
hei miten menee
ola tudo bom
salut, ça roule?
namaste, kaise ho
merhaba nasılsın
hola hola, todo bien?
hej, hur är läget
ahoj, jak se máš
γειά, τι κάνεις
""".strip().split("\n")

prompt = prompt.replace("%README%", readme)

# DeepSeek JSON Output mode (guarantees valid JSON string)
response_format = {"type": "json_object"}

# Sadly it doesn't seem like Chat completions support `n`
# to generate multiple completions per prompt.
base_payload = {
  # DeepSeek-V3.2 (non-thinking). Use "deepseek-reasoner" or thinking={"type":"enabled"} for thinking mode.
  "model": "deepseek-chat",  
  "stream": False,
  "response_format": response_format,
  "temperature": 1.0,
  "max_tokens": 4096, # Set reasonably to avoid truncating JSON.
}

def generate_conversation(idx: int, use_personas: bool = False, max_retries: int = 3):
    """
    Generate a single conversation using the DeepSeek API.
    Returns a list of message dicts with 'role' and 'content' keys.
    Includes retry logic with exponential backoff for robustness.
    """

    rng = random.Random(idx) # use idx as seed to the rng
    
    # Pick 5 example user first messages for inspiration
    user_first_prompt = "\n".join(rng.choice(user_first_prompts) for _ in range(5))
    
    payload = copy.deepcopy(base_payload)
    
    modified_prompt = prompt.replace("%USER_FIRST_PROMPTS%", user_first_prompt)
    
    if use_personas:
        persona = rng.choice(PERSONAS)
        persona_section = f"You are simulating a conversation where the User has a specific persona.\nPERSONA: {persona}"
        persona_role = " (embodying the persona above)"
        persona_instruction = "The User should ask questions or make comments consistent with their persona (e.g., specific vocabulary, interests, skepticism, or confusion)."
    else:
        persona_section = ""
        persona_role = ""
        persona_instruction = "The User should sound like a curious developer or enthusiast."

    modified_prompt = modified_prompt.replace("%PERSONA_SECTION%", persona_section)
    modified_prompt = modified_prompt.replace("%PERSONA_ROLE%", persona_role)
    modified_prompt = modified_prompt.replace("%PERSONA_INSTRUCTION%", persona_instruction)
    
    payload['messages'] = [{"role": "user", "content": modified_prompt}]

    last_exception = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            if not content or not content.strip():
                # DeepSeek docs note JSON Output may occasionally return empty content.
                raise RuntimeError(f"Empty model content. Full response: {result}")

            # Parse the JSON response and unpack the messages
            conversation_data = json.loads(content)
            messages = conversation_data['messages']
            
            return messages

        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s...
                time.sleep(2 ** attempt)
            else:
                raise last_exception

def calculate_diversity_metrics(conversations):
    """
    Calculate and print enhanced diversity metrics.
    """
    print("\n--- Diversity Metrics ---")
    ngram_counts = Counter()
    total_ngrams = 0
    all_words = []
    message_lengths = [] # in words
    turn_counts = []
    
    for msgs in conversations:
        # Filter for content messages
        content_msgs = [m['content'] for m in msgs if m.get('content')]
        turn_counts.append(len(content_msgs))
        
        text = " ".join(content_msgs)
        words = text.lower().split()
        all_words.extend(words)
        message_lengths.append(len(words))
        
        # 3-grams
        if len(words) >= 3:
            for i in range(len(words) - 2):
                ngram = tuple(words[i:i+3])
                ngram_counts[ngram] += 1
                total_ngrams += 1
            
    unique_ngrams = len(ngram_counts)
    diversity_score = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    unique_words = len(set(all_words))
    avg_words_per_conv = statistics.mean(message_lengths) if message_lengths else 0
    avg_turns = statistics.mean(turn_counts) if turn_counts else 0

    print(f"Total Conversations: {len(conversations)}")
    print(f"Avg Turns per Conversation: {avg_turns:.2f}")
    print(f"Avg Words per Conversation: {avg_words_per_conv:.2f}")
    print(f"Total Unique Words (Vocab): {unique_words}")
    print(f"Total Trigrams: {total_ngrams}")
    print(f"Unique Trigrams: {unique_ngrams}")
    print(f"Diversity Score (Unique/Total Trigrams): {diversity_score:.4f}")
    
    # Also print top 10 most common trigrams to see if there's repetition
    print("\nTop 10 Most Common Trigrams:")
    for ngram, count in ngram_counts.most_common(10):
        print(f"  {' '.join(ngram)}: {count}")
    print("-------------------------\n")


if __name__ == "__main__":
    
    # Default data directory
    DATA_DIR = os.path.join("edu", "gigachat", "data")
    
    parser = argparse.ArgumentParser(description="Generate synthetic conversation data for nanochat.")
    parser.add_argument("--use-personas", action="store_true", help="Enable persona-driven generation for higher diversity.")
    parser.add_argument("--num-conversations", type=int, default=20, help="Number of conversations to generate.")
    parser.add_argument("--num-workers", type=int, default=40, help="Number of parallel workers.")
    parser.add_argument("--output", type=str, default="identity_conversations.jsonl", help="Output JSONL filename or path.")
    
    args = parser.parse_args()

    # Configuration
    num_conversations = args.num_conversations
    num_workers = args.num_workers
    use_personas = args.use_personas
    
    # Ensure extension is .jsonl
    if not args.output.endswith(".jsonl"):
        args.output += ".jsonl"
    
    # Smart output path handling
    if os.path.dirname(args.output):
        # User provided a path with directory components, use as is
        output_file = args.output
    else:
        # User provided just a filename, save to default data dir
        output_file = os.path.join(DATA_DIR, args.output)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Starting generation with settings:")
    print(f"  Conversations: {num_conversations}")
    print(f"  Workers: {num_workers}")
    print(f"  Personas: {'Enabled' if use_personas else 'Disabled'}")
    print(f"  Output: {output_file}")

    # Wipe the file clean first if it exists (or append? Script implied wipe before)
    # The original script wiped it, so let's stick to that but maybe warn? 
    # For now, sticking to original behavior of overwriting run-specific output.
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed existing {output_file}")
    
    # Use ThreadPoolExecutor to generate conversations in parallel
    error_count = 0
    all_conversations = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(generate_conversation, idx, use_personas) for idx in range(num_conversations)]

        # Process results as they complete with tqdm progress bar
        for future in tqdm(as_completed(futures), total=num_conversations, desc="Generating conversations"):
            try:
                messages = future.result()

                # Lightly validate the conversation structure
                for i, message in enumerate(messages):
                    expected_role = "user" if i % 2 == 0 else "assistant"
                    assert message['role'] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"

                # If all looks good, write the messages to file
                with open(output_file, 'a') as f:
                    f.write(json.dumps(messages) + '\n')
                
                all_conversations.append(messages)

            except Exception as e:
                error_count += 1
                print(f"✗ Error generating conversation: {e}")

    print(f"\nDone! Successfully saved {len(all_conversations)} conversations to {output_file}")
    if error_count > 0:
        print(f"Encountered {error_count} errors during generation")

    if all_conversations:
        calculate_diversity_metrics(all_conversations)
