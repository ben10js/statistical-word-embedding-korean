import json
import os
from src.config import CONFIG

class BridgeManager:
    def __init__(self):
        # Ensure data directory exists
        self.bridge_path = os.path.join(CONFIG.get("data_dir", "data"), "bridge_corpus.json")
        self.bridge_data = self.load_bridge()

    def load_bridge(self):
        if os.path.exists(self.bridge_path):
            try:
                with open(self.bridge_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Bridge] Load error: {e}")
                return {}
        return {}

    def save_bridge(self):
        try:
            with open(self.bridge_path, 'w', encoding='utf-8') as f:
                json.dump(self.bridge_data, f, ensure_ascii=False, indent=2)
            # print(f"[Bridge] Saved to {self.bridge_path}")
        except Exception as e:
            print(f"[Bridge] Save error: {e}")

    def add_mapping(self, unknown_query, selected_proxy):
        """
        Maps an unknown query to a known external term (proxy).
        """
        if unknown_query not in self.bridge_data:
            self.bridge_data[unknown_query] = []
        
        # Avoid duplicates
        if selected_proxy not in self.bridge_data[unknown_query]:
            self.bridge_data[unknown_query].append(selected_proxy)
            self.save_bridge()
            return True
        return False

    def get_proxies(self, query):
        """
        Returns list of proxy words for a given query.
        """
        return self.bridge_data.get(query, [])
