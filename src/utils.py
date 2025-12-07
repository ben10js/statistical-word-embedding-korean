
from collections import Counter
import numpy as np
import json
import os
from .config import CONFIG

def count_word_frequencies(tokenized_sentences):
    """
    tokenized_sentences: List of token lists
    return: word_counts (Counter)
    """
    word_counts = Counter()
    for sent in tokenized_sentences:
        word_counts.update(sent)
    return word_counts

def top_k_words(word_counts, k=20):
    return [w for w, _ in word_counts.most_common(k)]

def load_external_embeddings():
    """
    Load embeddings and vocab from paths defined in config
    """
    # Assuming external files are named similarly or passed as args
    # For now, keeping legacy behavior but using config if applicable or relative paths
    ext_emb_path = 'external_embeddings_ppmi_svd.npy'
    ext_vocab_path = 'external_vocab_index.json'
    
    try:
        if os.path.exists(ext_emb_path):
             external_embeddings = np.load(ext_emb_path)
             with open(ext_vocab_path, 'r', encoding='utf-8') as f:
                 external_vocab = json.load(f)
             return external_embeddings, external_vocab
        else:
             return None, None
    except Exception:
        return None, None
