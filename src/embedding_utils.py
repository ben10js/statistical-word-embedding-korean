
from collections import Counter, defaultdict
from scipy import sparse
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def build_cooccurrence(sent_tokens, window_size, min_count):
    # 1) 전체 빈도 계산
    counter = Counter()
    for sent_idx, sent in enumerate(sent_tokens):
        counter.update(sent)
        if (sent_idx + 1) % 1000 == 0:
            print(f"[build_cooccurrence] Counting freq: {sent_idx+1}/{len(sent_tokens)}")

    # 2) min_count 이상 단어만 필터
    vocab_items = [w for w,c in counter.items() if c >= min_count]
    vocab = {w:i for i,w in enumerate(vocab_items)}
    V = len(vocab)
    print(f"[INFO] vocab size after min_count={min_count}: {V}")

    # 3) 공기카운트 집계 (인덱스 기반)
    cooc = defaultdict(Counter)
    for sent_idx, sent in enumerate(sent_tokens):
        L = len(sent)
        for idx, w in enumerate(sent):
            if w not in vocab: 
                continue
            wi = vocab[w]
            left = max(0, idx - window_size)
            right = min(L, idx + window_size + 1)
            for j in range(left, right):
                if j == idx: 
                    continue
                c = sent[j]
                if c not in vocab:
                    continue
                ci = vocab[c]
                cooc[wi][ci] += 1
        if (sent_idx + 1) % 1000 == 0:
            print(f"[build_cooccurrence] Co-occurrence: {sent_idx+1}/{len(sent_tokens)}")

    # 4) build sparse matrix (COO)
    rows = []
    cols = []
    data = []
    for i, ctr in cooc.items():
        for j, cnt in ctr.items():
            if i >= V or j >= V:
                continue
            rows.append(i); cols.append(j); data.append(cnt)

    print(f"[INFO] cooc matrix size: {len(rows)} nonzero")
    if len(rows) == 0:
        X = sparse.csr_matrix((V, V), dtype=np.int64)
    else:
        X = sparse.coo_matrix((data, (rows, cols)), shape=(V, V)).tocsr()
    return X, vocab

def compute_ppmi(X):
    X = X.astype(np.float64)
    total = X.sum()
    row_sums = np.array(X.sum(axis=1)).flatten()
    col_sums = np.array(X.sum(axis=0)).flatten()
    rows, cols = X.nonzero()
    data = []
    for idx, (r, c) in enumerate(zip(rows, cols)):
        p_wc = X[r, c] / total
        p_w = row_sums[r] / total
        p_c = col_sums[c] / total
        pmi = np.log(p_wc / (p_w * p_c + 1e-9) + 1e-9)
        data.append(max(pmi, 0.0))
        if (idx + 1) % 50000 == 0:
            print(f"[compute_ppmi] {idx+1}/{len(rows)}")
    M = sparse.csr_matrix((data, (rows, cols)), shape=X.shape)
    return M

def compute_embeddings(M, n_components=200):
    print(f"[{M.shape}] Computing SVD...")
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    U = svd.fit_transform(M)  # U: (V, k)
    embeddings = normalize(U)
    return embeddings

def l2_normalize_embeddings(embeddings):
    return normalize(embeddings, axis=1)

def build_inv_vocab(vocab_index):
    return {idx: w for w, idx in vocab_index.items()}

def nearest_by_index(idx, emb_norm, topk=10):
    sims = emb_norm @ emb_norm[idx]
    inds = np.argsort(-sims)[1:topk+1]
    return inds, sims[inds]

def nearest_by_word(word, emb_norm, vocab_index, inv_vocab=None, topk=10):
    if word not in vocab_index:
        raise KeyError(f"'{word}' not found in vocab.")
    idx = vocab_index[word]
    inds, sims = nearest_by_index(idx, emb_norm, topk=topk)
    if inv_vocab is None:
        inv_vocab = build_inv_vocab(vocab_index)
    return [(inv_vocab[i], float(s)) for i, s in zip(inds, sims)]
