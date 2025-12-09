# Statistical Word Embedding for Korean Metacognition

## Overview
This project implements a **Hybrid Vocabulary Bridging System** that augments a small-scale private corpus with large-scale pre-trained embeddings (FastText) to resolve Data Sparsity and Out-of-Vocabulary (OOV) issues. Designed for metacognitive analysis, it connects abstract psychological terms in personal notes to broader semantic concepts found in public knowledge bases. The system features a human-in-the-loop workflow where users explicitly sanction "bridge terms," ensuring the personal vector space is expanded accurately without semantic drift.

## Data
- **Internal Data**: Private Obsidian Markdown notes (Small, domain-specific).
- **External Data**: **Facebook FastText (cc.ko.300)** – purely data-driven, pre-trained on Common Crawl and Wikipedia (Large-scale).
- **Bridge Data**: User-curated JSON mapping (`bridge_corpus.json`) linkage between private and public embedding spaces.

## Methods
- **Embedding Alignment**:
  - **Internal**: Constructed using **PPMI (Positive Pointwise Mutual Information)** and **Truncated SVD** for efficient local semantic capture.
  - **External**: Integrated **FastText** (300-dim) to handle morphological nuances and OOV terms.
- **Search Algorithm**:
  - Implemented a widely expandable **Proxy Search** mechanism: *Query $\rightarrow$ External Neighbor $\rightarrow$ User Verification $\rightarrow$ Internal Re-query*.
- **Korean NLP**: Utilized **MeCab** for morphological segmentation and normalization.

## Results
- **Semantic Expansion**: Successfully retrieved relevant personal notes using query terms that never appeared in the text (e.g., searching "Retrospection" finds "Reflection" via external semantic similarity).
- **Precision**: The "Human-in-the-loop" bridging strategy maintained high retrieval relevance compared to fully automated query expansion which often introduces noise.
- **Scalability**: Capable of handling millions of external vectors efficiently alongside the lightweight internal model.

## How to run

### 1. Requirements
- Python 3.9+
- 4GB+ RAM (for FastText model)

### 2. Installation
```bash
git clone https://github.com/ben10js/statistical-word-embedding-korean.git
cd statistical-word-embedding-korean
pip install -r requirements.txt
```

### 3. Setup External Corpus
Download and convert the official FastText model (~1.2GB compressed):
```bash
python src/import_fasttext.py
```

### 4. Execution
Run the interactive search engine:
```bash
python interactive_search.py
```

## What I learned
- **Big Data Integration**: Learned practical strategies for integrating massive pre-trained models (GB-scale) with small, sparse local datasets.
- **Vector Space Dynamics**: Gained insight into how different embedding techniques (Count-based SVD vs. Prediction-based FastText) capture different aspects of semantic relationships.
- **OOV Resolution**: Understood that "Bridging" via a third-party corpus is a more robust solution for expanding limited vocabulary than simple synonym dictionaries.
