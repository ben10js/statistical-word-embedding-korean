import os
import gzip
import shutil
import urllib.request
import numpy as np
import json
import tqdm
import sys

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import CONFIG

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz"
DOWNLOAD_DIR = "downloads"
VEC_FILENAME = "cc.ko.300.vec"
GZ_FILENAME = "cc.ko.300.vec.gz"

def download_file(url, output_path):
    print(f"Downloading {url}...")
    with tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=output_path) as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)
        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)

def convert_fasttext_to_npy():
    # 1. Setup paths
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    gz_path = os.path.join(DOWNLOAD_DIR, GZ_FILENAME)
    vec_path = os.path.join(DOWNLOAD_DIR, VEC_FILENAME)
    
    # 2. Download if not exists
    if not os.path.exists(vec_path):
        if not os.path.exists(gz_path):
            print("FastText vector file not found. Downloading...")
            download_file(FASTTEXT_URL, gz_path)
            print("Download complete.")
        
        print("Extracting gzip...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(vec_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Extraction complete.")
    
    # 3. Parse and Convert
    print(f"Parsing {vec_path}...")
    vocab = {}
    embeddings = []
    
    # FastText .vec format: First line is "N D" (count dimension)
    # Subsequent lines: "word 0.1 0.2 ..."
    
    # We might limit vocab size to save memory (e.g., top 100k words)
    # FastText ko has 2M words, might be too big for local .npy without memmap.
    # Let's cap at 200,000 for efficiency in this project.
    MAX_VOCAB = 200000 
    
    with open(vec_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip().split()
        total_words = int(first_line[0])
        dim = int(first_line[1])
        print(f"FastText Source: {total_words} words, {dim} dimensions.")
        print(f"Importing top {MAX_VOCAB} words...")
        
        for i, line in tqdm.tqdm(enumerate(f), total=MAX_VOCAB):
            if i >= MAX_VOCAB:
                break
            parts = line.rstrip().split(' ')
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            
            # Skip if dim mismatch (some broken lines in crawls)
            if len(vector) != dim:
                continue
                
            vocab[word] = i
            embeddings.append(vector)
            
    embeddings_matrix = np.array(embeddings)
    
    # 4. Save to Project format
    output_emb_path = os.path.join(CONFIG["embedding_dir"], 'external_embeddings_ppmi_svd.npy') # Overwriting/Using logical name
    output_vocab_path = os.path.join(CONFIG["vocab_dir"], 'external_vocab_index.json')
    
    print(f"Saving to {output_emb_path}...")
    np.save(output_emb_path, embeddings_matrix)
    
    print(f"Saving vocab to {output_vocab_path}...")
    with open(output_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)
        
    print("\n[Success] External Corpus Upgraded to FastText!")
    print("Run 'python interactive_search.py' to test.")

if __name__ == "__main__":
    convert_fasttext_to_npy()
