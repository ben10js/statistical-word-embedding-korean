import numpy as np
import json

# 외부 임베딩 로드 (다음 실행시)
def load_external_embeddings():
    try:
        external_embeddings = np.load('external_embeddings_ppmi_svd.npy')
        with open('external_vocab_index.json', 'r', encoding='utf-8') as f:
            external_vocab = json.load(f)
        return external_embeddings, external_vocab
    except FileNotFoundError:
        return None, None
        