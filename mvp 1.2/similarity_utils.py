# similarity_utils.py
import json
import math
import pickle
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from rapidfuzz import process, fuzz
from collections import defaultdict

Word = str
Candidate = Tuple[Word, float]

# ------------------------
# 유틸: 저장/로드
# ------------------------
def save_pickle(obj, path):
    with open(path, "wb") as fw:
        pickle.dump(obj, fw)

def load_pickle(path):
    with open(path, "rb") as fr:
        return pickle.load(fr)

def save_json(obj, path):
    with open(path, "w", encoding="utf8") as fw:
        json.dump(obj, fw, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf8") as fr:
        return json.load(fr)

# ------------------------
# 임베딩 유틸
# ------------------------
def l2_normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """행 단위 L2 정규화. 반환 dtype은 float32 추천."""
    if emb is None:
        return None
    embf = emb.astype(np.float32)
    embn = normalize(embf, axis=1)
    return embn

# ------------------------
# nearest_by_word (embedding cosine)
# 내적으로 cosine을 계산 (입력 emb는 L2 정규화되어야 함)
# ------------------------
def nearest_by_word(query_word: Word, emb_norm: np.ndarray, vocab: Dict[Word,int], topk:int=10) -> List[Candidate]:
    if emb_norm is None or vocab is None:
        return []
    if query_word not in vocab:
        return []
    inv = {i:w for w,i in vocab.items()}
    qidx = vocab[query_word]
    qvec = emb_norm[qidx]
    sims = emb_norm.dot(qvec)
    sims[qidx] = -np.inf
    idxs = np.argsort(-sims)[:topk]
    return [(inv[i], float(sims[i])) for i in idxs]

# ------------------------
# nearest_by_pmi
# pmi_matrix: scipy.sparse or numpy array (V,V)
# ------------------------
def nearest_by_pmi(query_word: Word, pmi_matrix, vocab: Dict[Word,int], topk:int=10) -> List[Candidate]:
    if pmi_matrix is None or vocab is None or query_word not in vocab:
        return []
    inv = {i:w for w,i in vocab.items()}
    qidx = vocab[query_word]
    if sparse.issparse(pmi_matrix):
        row = pmi_matrix.getrow(qidx).toarray().ravel()
    else:
        row = np.array(pmi_matrix[qidx], dtype=float)
    row[qidx] = -np.inf
    idxs = np.argsort(-row)[:topk]
    return [(inv[i], float(row[i])) for i in idxs if not math.isinf(row[i])]

# ------------------------
# ngram / Jaccard 관련 유틸
# ------------------------
def char_ngrams(word: str, nmin:int=3, nmax:int=6) -> Set[str]:
    s = f"<{word}>"
    L = len(s)
    out = set()
    for n in range(nmin, min(nmax, L)+1):
        for i in range(0, L-n+1):
            out.add(s[i:i+n])
    return out

def build_word_to_ngrams(vocab: Dict[Word,int], nmin:int=3, nmax:int=6):
    """vocab: word->idx. 반환 word_to_ngrams(word)->set(ngrams) 및 idx->ngrams."""
    word_to_ngrams = {}
    idx_to_ngrams = {}
    for w, idx in vocab.items():
        grams = char_ngrams(w, nmin=nmin, nmax=nmax)
        word_to_ngrams[w] = grams
        idx_to_ngrams[idx] = grams
    return word_to_ngrams, idx_to_ngrams

def build_ngram_index(word_to_ngrams: Dict[Word, Set[str]]):
    """ngram_index: ngram -> set(word_idx) requires word->idx available separately."""
    ngram_index = defaultdict(set)
    # We expect caller to also have a vocab dict to map words->idx
    # But here we assume word_to_ngrams keys are words and caller will map words->idx externally.
    for w, grams in word_to_ngrams.items():
        # word_to_ngrams should be built with knowledge of idx; but if not, leave as word-based
        for g in grams:
            ngram_index[g].add(w)
    return dict(ngram_index)

def build_ngram_index_from_vocab(vocab: Dict[Word,int], nmin:int=3, nmax:int=6):
    """더 명시적: ngram -> set(idx)"""
    ngram_index = defaultdict(set)
    for w, idx in vocab.items():
        grams = char_ngrams(w, nmin=nmin, nmax=nmax)
        for g in grams:
            ngram_index[g].add(idx)
    return dict(ngram_index)

# ------------------------
# nearest_by_jaccard (idx 기반 인덱스 권장)
# ------------------------
def nearest_by_jaccard(query_word: Word,
                       ngram_index: Optional[Dict[str, Set[int]]],
                       vocab: Dict[Word,int],
                       topk:int=10,
                       word_to_ngrams: Optional[Dict[Word,Set[str]]] = None) -> List[Candidate]:
    if vocab is None:
        return []
    inv = {i:w for w,i in vocab.items()}
    # q grams
    qgrams = word_to_ngrams.get(query_word, char_ngrams(query_word)) if word_to_ngrams else char_ngrams(query_word)
    cand_idxs = set()
    if ngram_index is not None:
        for g in qgrams:
            if g in ngram_index:
                cand_idxs.update(ngram_index[g])
    else:
        # fallback: all vocabulary
        cand_idxs = set(range(len(vocab)))
    # compute exact jaccard using idx->grams if available
    # try to use word_to_ngrams mapping by word; if it is word->grams mapping, we need idx->word
    # we'll attempt to use provided word_to_ngrams (word->grams) to find grams by idx; else fallback approximate
    idx_to_grams = {}
    if word_to_ngrams:
        # build idx->grams
        for w, grams in word_to_ngrams.items():
            if w in vocab:
                idx_to_grams[vocab[w]] = grams
    # if idx_to_grams incomplete, fallback to approximate (shared count / |qgrams|)
    scores = []
    for idx in cand_idxs:
        grams = idx_to_grams.get(idx, None)
        if grams is not None:
            inter = len(qgrams & grams)
            union = len(qgrams | grams)
            score = inter/union if union>0 else 0.0
        else:
            # approximate: count shared via ngram_index membership
            shared = 0
            for g in qgrams:
                if g in ngram_index and idx in ngram_index[g]:
                    shared += 1
            score = shared / max(1, len(qgrams))
        scores.append((idx, score))
    scores.sort(key=lambda x: -x[1])
    out = [(inv[idx], float(sc)) for idx, sc in scores[:topk]]
    return out

# ------------------------
# nearest_by_edit_distance (RapidFuzz)
# ------------------------
def nearest_by_edit_distance(query_word: Word, vocab: Dict[Word,int], topk:int=10) -> List[Candidate]:
    if vocab is None:
        return []
    vocab_list = list(vocab.keys())
    results = process.extract(query_word, vocab_list, scorer=fuzz.WRatio, limit=topk)
    return [(match, score/100.0) for match, score, _ in results]

# ------------------------
# nearest_by_tfidf
# tfidf_matrix: scipy.sparse or numpy array; orientation auto-detected
# ------------------------
def nearest_by_tfidf(query_word: Word, tfidf_matrix, vocab: Dict[Word,int], topk:int=10) -> List[Candidate]:
    if tfidf_matrix is None or vocab is None or query_word not in vocab:
        return []
    inv = {i:w for w,i in vocab.items()}
    j = vocab[query_word]
    # ensure numpy or sparse
    if sparse.issparse(tfidf_matrix):
        shape = tfidf_matrix.shape
    else:
        tfidf_matrix = np.asarray(tfidf_matrix)
        shape = tfidf_matrix.shape
    # (n_docs, V) case
    if shape[0] < shape[1]:
        X = tfidf_matrix
        if sparse.issparse(X):
            col = X.getcol(j)
            numer = (X.T).dot(col).toarray().ravel()
            col_norm = math.sqrt(col.multiply(col).sum())
            col_sq = np.array(X.power(2).sum(axis=0)).ravel()
            denom = np.sqrt(col_sq) * (col_norm if col_norm>0 else 1e-12)
            scores = numer / (denom + 1e-12)
        else:
            col = X[:, j]
            norms = np.linalg.norm(X, axis=0) + 1e-12
            numer = X.T.dot(col)
            denom = norms * (np.linalg.norm(col) + 1e-12)
            scores = numer / denom
        scores[j] = -np.inf
        idxs = np.argsort(-scores)[:topk]
        return [(inv[i], float(scores[i])) for i in idxs]
    else:
        X = tfidf_matrix
        if sparse.issparse(X):
            row = X.getrow(j)
            numer = X.dot(row.T).toarray().ravel()
            row_norm = math.sqrt(row.multiply(row).sum())
            row_sq = np.array(X.power(2).sum(axis=1)).ravel()
            denom = np.sqrt(row_sq) * (row_norm if row_norm>0 else 1e-12)
            scores = numer / (denom + 1e-12)
        else:
            row = X[j]
            norms = np.linalg.norm(X, axis=1) + 1e-12
            numer = X.dot(row)
            denom = norms * (np.linalg.norm(row) + 1e-12)
            scores = numer / denom
        scores[j] = -np.inf
        idxs = np.argsort(-scores)[:topk]
        return [(inv[i], float(scores[i])) for i in idxs]

# ------------------------
# aggregate_candidates
# normalize each method's scores via min-max and weighted sum
# ------------------------
def aggregate_candidates(method_results: Dict[str, List[Candidate]],
                         weights: Optional[Dict[str,float]] = None,
                         topk: int = 20) -> List[Tuple[Word, float]]:
    if weights is None:
        weights = {'embedding':1.0, 'pmi':0.8, 'jaccard':0.5, 'edit_distance':0.3, 'tfidf':0.9}
    # union
    cand_set = set()
    mapping = {}
    for m, lst in method_results.items():
        mapping[m] = {w:s for w,s in lst}
        cand_set.update([w for w,_ in lst])
    cand_list = list(cand_set)
    # per-method min-max
    norm_scores = {m:{} for m in mapping}
    for m, mmap in mapping.items():
        vals = np.array([mmap.get(w, 0.0) for w in cand_list], dtype=float)
        vmin = vals.min() if vals.size>0 else 0.0
        vmax = vals.max() if vals.size>0 else 1.0
        if vmax - vmin < 1e-12:
            for i,w in enumerate(cand_list):
                norm_scores[m][w] = 1.0 if vals[i] > 0 else 0.0
        else:
            for i,w in enumerate(cand_list):
                norm_scores[m][w] = float((vals[i]-vmin)/(vmax-vmin))
    # combine
    combined = []
    for w in cand_list:
        s = 0.0; wsum = 0.0
        for m in norm_scores:
            wt = weights.get(m, 1.0)
            s += wt * norm_scores[m].get(w, 0.0)
            wsum += wt
        combined.append((w, s/(wsum+1e-12)))
    combined.sort(key=lambda x: -x[1])
    return combined[:topk]