import numpy as np
from sklearn.preprocessing import normalize

# --- 1) L2 정규화 함수 ---
def l2_normalize_embeddings(embeddings):
    """embeddings 행렬을 L2 정규화"""
    return normalize(embeddings, axis=1)

# --- 2) 역인덱스 생성 함수 ---
def build_inv_vocab(vocab_index):
    """
    vocab_index: {word: idx} 형태 딕셔너리
    return: {idx: word} 역인덱스 딕셔너리
    """
    return {idx: w for w, idx in vocab_index.items()}

# --- 3) nearest neighbor by index ---
def nearest_by_index(idx, emb_norm, topk=10):
    sims = emb_norm @ emb_norm[idx]
    inds = np.argsort(-sims)[1:topk+1]
    return inds, sims[inds]

# --- 4) nearest neighbor by word ---
def nearest_by_word(word, emb_norm, vocab_index, inv_vocab=None, topk=10):
    if word not in vocab_index:
        raise KeyError(f"'{word}' not found in vocab.")
    idx = vocab_index[word]
    inds, sims = nearest_by_index(idx, emb_norm, topk=topk)
    # inv_vocab을 argument로 전달하거나 함수 내에서 생성 가능
    if inv_vocab is None:
        inv_vocab = build_inv_vocab(vocab_index)
    return [(inv_vocab[i], float(s)) for i, s in zip(inds, sims)]
