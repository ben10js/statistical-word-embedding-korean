print("=== SCRIPT STARTING ===")

import traceback
import pickle
import os
import numpy as np
import json
from tqdm import tqdm

# Config
from src.config import CONFIG

# Modules
from src.processed import doc_to_sent_tokens
from src.text_utils import filter_sentences, StopwordFilter
from src.embedding_utils import build_cooccurrence, compute_ppmi, compute_embeddings, l2_normalize_embeddings, build_inv_vocab, nearest_by_word
from src.preprocess import load_obsidian_notes
# from src.check_wikicache import load_external_embeddings as check_wiki_load # Consolidated into src.utils
from src.preprocess_wiki import load_wiki_corpus
from src.expand_vocabulary import expand_word_pool_iterative

def main():
    # Paths from Config
    embedding_path = os.path.join(CONFIG["embedding_dir"], CONFIG["embedding_file"])
    vocab_index_path = os.path.join(CONFIG["vocab_dir"], CONFIG["vocab_index_file"])

    # 1. Main Pipeline
    if os.path.exists(embedding_path) and os.path.exists(vocab_index_path):
        print("Cache exists! Skipping pipeline.")
        embeddings = np.load(embedding_path)
        with open(vocab_index_path, 'r', encoding='utf-8') as f:
            vocab_index = json.load(f)
        print(f"Cached embeddings shape: {embeddings.shape}")
    else:
        try:
            print("Entering main pipeline...")
            
            # 1. Load Documents
            vault = CONFIG["vault_path"]
            docs = load_obsidian_notes(vault)
            print(f"Loaded {len(docs)} documents.")

            # 2. Tokenize
            corpus_sent_tokens = []
            for d in docs:
                sents = doc_to_sent_tokens(d["text"])
                corpus_sent_tokens.extend(sents)
            print(f"Total {len(corpus_sent_tokens)} sentences.")

            # 3. Filter
            filtered_sentences = filter_sentences(corpus_sent_tokens)
            main_filterer = StopwordFilter()
            filtered_sentences = main_filterer.filter(filtered_sentences)
            print(f"Filtered sentences count: {len(filtered_sentences)}")

            # 4. PPMI
            X, vocab_index = build_cooccurrence(filtered_sentences, window_size=CONFIG["window_size"], min_count=CONFIG["min_count"])
            M = compute_ppmi(X)
            print(f"PPMI shape: {M.shape}")

            # 5. SVD
            embeddings = compute_embeddings(M, n_components=CONFIG["svd_components"])
            print(f"Embeddings shape: {embeddings.shape}")

            # 6. Save
            np.save(embedding_path, embeddings)
            with open(vocab_index_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_index, f, ensure_ascii=False, indent=2)
            print("Embeddings and vocab_index saved!")

        except Exception as e:
            print("Error in pipeline:")
            traceback.print_exc()
            return

    # 4. Analysis (Optional debug)
    # word_counts = count_word_frequencies(filtered_sentences) ...

    # 5. Interactive Search Suggestion
    print("\nRun 'python interactive_search.py' to use the search system.")

if __name__ == "__main__":
    main()

   
 

   
    
