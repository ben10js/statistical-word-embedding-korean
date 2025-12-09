# Statistical Word Embedding for Korean Metacognition

## Overview
This project implements a statistical word embedding model from scratch to analyze and expand the vocabulary related to "metacognition" in the Korean language. Unlike pre-trained heavy models (like BERT), this lightweight approach builds a domain-specific vector space using Co-occurrence Matrices, PPMI (Positive Pointwise Mutual Information), and SVD (Singular Value Decomposition). It is designed to capture subtle semantic nuances in abstract psychological terms.

## Data
- **Corpus**: Korean Wikipedia (Dump), NamuWiki (selected subsets), and specialized psychology texts.
- **Dictionary**: MeCab-ko dictionary for precise Korean morphological analysis.
- **Format**: Plain text files processed into tokenized streams.

## Methods
- **Morphological Analysis**: Utilized **MeCab** (via `konlpy`/`ko-dic`) to handle Korean agglutinative structure, extracting nouns and significant roots.
- **Statistical Modeling**:
  - **Co-occurrence Matrix**: Constructed a symmetric matrix counting word neighbors within a fixed window size.
  - **PPMI weighting**: Applied Positive Pointwise Mutual Information to normalize frequencies and highlight meaningful associations over common ones.
  - **Dimensionality Reduction (SVD)**: Applied Truncated SVD to reduce the sparse high-dimensional matrix into dense word vectors (size: 100-300).
- **Expansion**: Used Cosine Similarity to retrieve nearest neighbors for seed terms like "introspection" or "self-regulation".

## Results
- **Domain-Specific Embeddings**: Successfully clustered abstract concepts (e.g., '성찰'(reflection) $\approx$ '회고'(retrospection)) more effectively for this specific domain than general-purpose small embeddings.
- **Efficiency**: The entire pipeline runs on specific CPUs without requiring GPU acceleration, creating a portable and reproducible NLP module.
- **Visualization**: Capable of projecting high-dimensional word relationships into 2D space for analysis.

## How to run

### 1. Installation
```bash
git clone https://github.com/ben10js/statistical-word-embedding-korean.git
cd statistical-word-embedding-korean
pip install -r requirements.txt
```

### 2. Execution
**Build Embeddings**:
```bash
python main.py
```
*This tokenizes the corpus, builds the matrix, and saves the vectors to `data/`.*

**Interactive Search**:
```bash
python interactive_search.py
```
*Enter a query word to see its nearest semantic neighbors.*

## What I learned
- **Matrix Operations**: Gained deep understanding of linear algebra in NLP, specifically how SVD compresses semantic information.
- **Korean NLP Challenges**: Learned to tackle unique challenges in Korean tokenization (postposition removal, compound nouns) which are absent in English NLP.
- **Sparse Data Handling**: Implemented optimized sparse matrix operations (`scipy.sparse`) to handle vocabulary sizes exceeding 100,000 terms efficiently.