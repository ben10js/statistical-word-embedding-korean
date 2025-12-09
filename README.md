# Statistical Word Embedding for Pure Personal Retrieval

## Overview
This project is an experiment in **"Zero-Bias" Information Retrieval** designed to explore a personal knowledge base (Obsidian) without the influence of massive, pre-trained external models (like GPT or BERT). It builds a custom, lightweight semantic space purely from the user's own writing using classical statistical methods (**Co-occurrence Matrix + PPMI + SVD**). The goal is to capture the unique, idiosyncratic associations within a personal corpus—even if the resulting vectors are "rough"—to provide authentic retrospective retrieval.

## Data
- **Corpus**: Personal Obsidian Markdown notes.
- **Why Internal Only?**: To prevent "semantic contamination" from public datasets. If the user associates "Apple" with "Design" rather than "Fruit," the model should reflect strictly that personal usage.

## Methods (Pure Statistical Approach)
- **Tokenization**: **MeCab** (Korean) is used for basic morphological analysis, deliberately avoiding complex sub-word tokenizers to maintain word-level interpretability.
- **Vector Space Construction**:
  1.  **Co-occurrence Matrix**: Counts word neighbors within a fixed window (Window size: 5-10).
  2.  **PPMI (Positive Pointwise Mutual Information)**: Weighs the informativeness of co-occurrences, filtering out random noise.
  3.  **Dimensionality Reduction (SVD)**: Applies **Truncated SVD** to compress the sparse matrix into dense vectors (Dim: 100-200), capturing latent semantic structures purely from internal data.
- **OOV Handling (Optional Add-on)**: Only when a query is completely absent from the internal space, an **External Bridge (FastText)** is consulted to find a "proxy word" that *does* exist internally, preserving the user-centric search loop.

## Results
- **Authentic Associations**: Successfully retrieves notes based on the user's specific context (e.g., retrieving project notes via abstract concepts like "Structure" or "Flow") rather than general dictionary definitions.
- **Transparency**: Unlike black-box neural networks, the retrieval logic is mathematically transparent (Linear Algebra), allowing full traceability of why two words are considered similar.
- **Efficiency**: The entire indexing and retrieval pipeline runs instantly on a standard CPU.

## How to run

### 1. Installation
```bash
git clone https://github.com/ben10js/statistical-word-embedding-korean.git
cd statistical-word-embedding-korean
pip install -r requirements.txt
```

### 2. Execution (Build Internal Index)
```bash
python main.py
```
*This scans your local vault and builds the SVD-based vector space from scratch.*

### 3. Search
```bash
python interactive_search.py
```

## What I learned
- **Power of Classical NLP**: Learned that for highly specialized or personal domains, simple statistical methods (SVD) can sometimes outperform generic large models by strictly adhering to the specific domain distribution.
- **Sparsity Challenges**: Encountered the limits of word-level embedding on small data (high sparsity) and learned to mitigate it via careful window-size tuning and PPMI weighting.
- **Value of "Rough" Data**: Realized that even "imperfect" embeddings can be valuable tools for metacognition, acting as a mirror to one's own writing habits and latent connections.
