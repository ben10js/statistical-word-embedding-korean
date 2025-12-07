
# MetaCogAI via HyperCLOVA

**MetaCogAI** is a semantic network analysis tool that analyzes personal knowledge bases (e.g., Obsidian vaults) and connects them with external knowledge (Wikipedia/HyperCLOVA) to find semantic gaps and bridging concepts.

## Features
- **Semantic Indexing**: Constructs a co-occurrence based PPMI-SVD embedding space from Markdown notes.
- **Cross-Corpus Search**: Finds "bridging words" that connect personal notes with external concepts (using a Wikipedia derived corpus).
- **Interactive Feedback**: A CLI tool (`interactive_search.py`) to explore valid connections.

## Directory Structure
```text
metacogai-hyperclova/
├── data/               # Embeddings and vocab (ignored by git)
├── src/                # Core logic
│   ├── embedding_utils.py  # PPMI, SVD, Normalization
│   ├── text_utils.py       # Token cleaning, Stopwords
│   ├── processed.py        # Spacy Tokenization
│   ├── preprocess.py       # Obsidian file loading
│   └── config.py           # Paths and settings
├── main.py             # Main indexing pipeline
├── interactive_search.py # Search UI
└── requirements.txt
```

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. **Config**:
   - Edit `src/config.py` to point to your Obsidian vault path.

3. **Build Index**:
   ```bash
   python main.py
   ```
   This will generate embeddings in `data/embeddings/`.

4. **Search**:
   ```bash
   python interactive_search.py
   ```

## Requirements
- Python 3.10+
- `mecab-python3` (Windows wheel may be required)