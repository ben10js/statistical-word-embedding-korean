# Statistical Word Embedding for Korean Metacognition

## Overview
This project implements a **Vocabulary Bridging System** that connects a small internal corpus (private notes) with a large external corpus (Wikipedia/NamuWiki) to expand the semantic range of metacognitive terms. It addresses the "Out-of-Vocabulary (OOV)" problem in personal datasets by selectively importing related concepts from a broader knowledge base. When a term is missing internally, the system queries the external embedding model and, upon user approval, registers semantic neighbors into a "Bridge Corpus" for future analysis.

## Data
- **Internal Corpus**: Small-scale private notes (e.g., Obsidian vault).
- **External Corpus**: Large-scale Korean text (Wikipedia, NamuWiki) used as a fallback knowledge base.
- **Bridge Corpus**: A curated collection of terms added by the user to bridge the gap between internal and external knowledge.

## Methods
- **Hybrid Search Strategy**:
  1.  **Internal Lookup**: Initially attempts to retrieve nearest neighbors (Top-K) from the internal corpus.
  2.  **External Fallback**: If the query is absent internally, the system searches the external corpus to find semantically similar candidate terms.
  3.  **User-Driven Bridging**: The system presents these external candidates to the user. The user selects the term that best matches their intended nuance, which is then registered in the **"Bridge Corpus."**
  4.  **Proxy Search (Re-entry)**: Once bridged, the system can use this approved external term as a **semantic proxy** to search the internal corpus again. This effectively connects a previously unknown query to relevant existing notes via the user-selected bridge term.
- **Statistical Modeling**:
  - Uses **PPMI (Positive Pointwise Mutual Information)** and **SVD (Singular Value Decomposition)** to build lightweight, distinct vector spaces for both corpora.
- **Korean Processing**: Utilizes **MeCab** for precise morphological analysis to handle agglutinative traits.

## Results
- **Vocabulary Expansion**: Successfully enables the system to "learn" new concepts (e.g., specific psychological terms) that were never explicitly written in the private notes but are relevant to the user's thinking.
- **Selective Knowledge Integration**: Avoids polluting the personal space with irrelevant external data by requiring explicit user confirmation for bridged terms.
- **Zero-GPU Efficiency**: Runs entirely on CPU using efficient sparse matrix operations (`scipy.sparse`), making it suitable for local environments.

## How to run

### 1. Installation
```bash
git clone https://github.com/ben10js/statistical-word-embedding-korean.git
cd statistical-word-embedding-korean
pip install -r requirements.txt
```

### 2. Execution
**Run the Interactive Bridge**:
```bash
python interactive_search.py
```
*Enter a query. If it's missing from your notes, the system will offer to fetch related terms from the external corpus.*

## What I learned
- **Handling Data Sparsity**: Learned that small personal datasets often lack sufficient context for robust embeddings, and bridging them with a larger corpus is a practical solution.
- **Interactive AI Design**: Designed a "Human-in-the-loop" workflow where the user acts as a filter, ensuring only high-quality, relevant terms are merged into the personal dataset.
- **Dual-Corpus Architecture**: Gained experience in managing and querying two distinct vector spaces simultaneously to augment limited data.
