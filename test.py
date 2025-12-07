from preprocess import load_obsidian_notes
print("preprocess imported")
        
vault = "C://Users//User//OneDrive - konkuk.ac.kr//문서//n8n_metacog"
ver = load_obsidian_notes(vault)
print(f"Loaded {len(docs)} documents.")
