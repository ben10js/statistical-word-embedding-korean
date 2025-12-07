from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Truncated SVD
def compute_embeddings(M, n_components=200):
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    U = svd.fit_transform(M)  # U: (V, k)
    # optionally scale by sqrt(Sigma): sklearn TruncatedSVD returns transformed matrix already
    embeddings = normalize(U)
    return embeddings
    

