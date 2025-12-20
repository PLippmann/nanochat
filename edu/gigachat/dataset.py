# load dataset from HF with appropriate caching and sharding
# link for first shard at: https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/blob/main/shard_00000.parquet
import os
import requests
import time
from multiprocessing import Pool
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/" # shard_00000.parquet
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
SAVE_LOCATION = "./edu/gigachat/data/"
MAX_RETRIES = 5
NUM_WORKERS = 8

def load_shard(shard_id):
    # Check if file exists, if not download it with backoff
    #print(f"Downloading shard {shard_id}")
    destination = SAVE_LOCATION + "shard_" + f"{shard_id:05d}" + ".parquet" # ShardID starts with 00000
    temp_destination = destination + ".tmp"

    if os.path.exists(destination):
        return True
    
    url = BASE_URL + "shard_" + f"{shard_id:05d}" + ".parquet"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            with open(temp_destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB 
                    if chunk:
                        f.write(chunk)
            os.rename(temp_destination, destination)
            #print(f"Downloaded shard {shard_id}. Success!")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if os.path.exists(temp_destination):
                os.remove(temp_destination)
            wait_time = 2 ** attempt
            time.sleep(wait_time)

    print(f"Failed to download shard {shard_id}")
    return False
    

if __name__ == "__main__":
    print("Downloading dataset...")
    # Check if directory exists, if not create it
    if not os.path.exists(SAVE_LOCATION):
        os.makedirs(SAVE_LOCATION, exist_ok=True)
    
    start_time = time.time()
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(load_shard, range(MAX_SHARD + 1)), # imap plays nicer with tqdm than map
            total=MAX_SHARD + 1, 
            desc="Downloading"
        ))
    end_time = time.time()

    print(f"Done! Downloaded {sum(results)} shards out of {MAX_SHARD + 1} in {(end_time - start_time) / 60:.2f} minutes.")