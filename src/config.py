
import os

CONFIG = {
    # Path to your Obsidian Vault or Note directory
    "vault_path": r"C:\Users\User\OneDrive - konkuk.ac.kr\문서\n8n_metacog",
    
    # Data directories
    "data_dir": "data",
    "embedding_dir": os.path.join("data", "embeddings"),
    "vocab_dir": os.path.join("data", "vocab"),
    
    # File names
    "embedding_file": "embeddings_ppmi_svd.npy",
    "vocab_index_file": "vocab_index.json",
    
    # Parameters
    "window_size": 4,
    "min_count": 5,
    "svd_components": 200
}

# Ensure directories exist
os.makedirs(CONFIG["embedding_dir"], exist_ok=True)
os.makedirs(CONFIG["vocab_dir"], exist_ok=True)
