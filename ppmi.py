from collections import Counter, defaultdict
from scipy import sparse
import numpy as np

def build_cooccurrence(sent_tokens, window_size, min_count):
    # 1) 전체 빈도 계산
    counter = Counter()
    for sent_idx, sent in enumerate(sent_tokens):
        counter.update(sent)
        print(f"[build_cooccurrence] Processing sentence {sent_idx}/{len(sent_tokens)}")

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
                # 디버깅 출력 (너무 많으면 주석처리)
                # print(f"[cooc] sent_idx={sent_idx}, w={w}, c={c}, wi={wi}, ci={ci}, count={cooc[wi][ci]}")
        if sent_idx % 100 == 0:
            print(f"[build_cooccurrence] Done sentence {sent_idx}/{len(sent_tokens)}")

    # 4) build sparse matrix (COO)
    rows = []
    cols = []
    data = []
    for i, ctr in cooc.items():
        for j, cnt in ctr.items():
            if i >= V or j >= V:
                print(f"[WARN] skipping out-of-range index i={i}, j={j}, V={V}")
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
        if idx % 10000 == 0:  # 대형 행렬이면 진행상황 출력
            print(f"[compute_ppmi] {idx}/{len(rows)}")
    M = sparse.csr_matrix((data, (rows, cols)), shape=X.shape)
    return M

# 예시 사용
# corpus_sent_tokens = [["나는", "나비다"], ["나비", "예쁘다"], ...]
# X, vocab_index = build_cooccurrence(corpus_sent_tokens, window_size=4, min_count=5)
# M = compute_ppmi(X)
