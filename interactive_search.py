import json
import numpy as np
import os
from difflib import get_close_matches

# Config
from src.config import CONFIG

# Modules
from src.embedding_utils import nearest_by_word, l2_normalize_embeddings

class MetacogAISearchUI:
    def __init__(self):
        self.main_embeddings = None
        self.main_vocab = None
        self.external_embeddings = None
        self.external_vocab = None

    def load_embeddings(self):
        # Paths from CONFIG
        main_emb_path = os.path.join(CONFIG["embedding_dir"], CONFIG["embedding_file"])
        main_vocab_path = os.path.join(CONFIG["vocab_dir"], CONFIG["vocab_index_file"])
        
        # External paths (assuming same structure or legacy names in root if not in config yet)
        # For now, let's assume they are in data/embeddings as well or use legacy names if they were generated there
        # But main.py saves to data/embeddings.
        # User might have external files in root?
        # Let's try to look in data/embeddings first, then root.
        
        ext_emb_path = os.path.join(CONFIG["embedding_dir"], 'external_embeddings_ppmi_svd.npy')
        ext_vocab_path = os.path.join(CONFIG["vocab_dir"], 'external_vocab_index.json')
        
        # Fallback to root if not found (legacy support)
        if not os.path.exists(ext_emb_path):
             ext_emb_path = 'external_embeddings_ppmi_svd.npy'
        if not os.path.exists(ext_vocab_path):
             ext_vocab_path = 'external_vocab_index.json'

        try:
            print(f"Loading main embeddings from {main_emb_path}...")
            self.main_embeddings = np.load(main_emb_path)
            with open(main_vocab_path, 'r', encoding='utf-8') as f:
                self.main_vocab = json.load(f)
            
            if os.path.exists(ext_emb_path):
                print(f"Loading external embeddings from {ext_emb_path}...")
                self.external_embeddings = np.load(ext_emb_path)
                with open(ext_vocab_path, 'r', encoding='utf-8') as f:
                    self.external_vocab = json.load(f)
                print(f"Loaded main vocab: {len(self.main_vocab)} / external vocab: {len(self.external_vocab)}")
            else:
                print("External embeddings not found. Only main corpus will be used.")
                self.external_vocab = {} # Empty to prevent errors
                
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
        return True

    def interactive_search(self):
        print("\n=== 메타인지 AI 단어 검색 시스템 ===")
        print("명령어: quit/exit/종료, help, 검색어 입력")
        while True:
            try:
                user_input = input("\n검색어를 입력하세요: ").strip()
                if user_input.lower() in ['quit', 'exit', '종료']:
                    print("검색을 종료합니다.")
                    break
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                if not user_input:
                    continue

                res = self.cross_corpus_search_with_feedback(user_input)
                self.display_results(user_input, res)
                if res["mode"] == "cross":
                    self.ask_feedback(user_input, res["found_words"])

            except KeyboardInterrupt:
                print("\n검색을 종료합니다.")
                break
            except Exception as e:
                print(f"오류가 발생했습니다: {e}")

    def cross_corpus_search_with_feedback(self, query_word, topk=10, expand_level=2, max_try_k=50):
        if query_word in self.main_vocab:
            # 메인코퍼스에 있으면 개인코퍼스 임베딩만!
            main_emb_norm = l2_normalize_embeddings(self.main_embeddings)
            main_results = nearest_by_word(query_word, main_emb_norm, self.main_vocab, topk=topk)
            return {"mode": "main", "main": main_results}
        else:
            if self.external_embeddings is None:
                 suggestions = get_close_matches(query_word, self.main_vocab.keys(), n=5)
                 return {"mode": "cross", "found_words": [], "suggestions": suggestions}

            ext_emb_norm = l2_normalize_embeddings(self.external_embeddings)
            found_words = []
            def get_candidates(word, k):
                if word in self.external_vocab:
                    return [w for w, _ in nearest_by_word(word, ext_emb_norm, self.external_vocab, topk=k)]
                return []

            # k 증분 확장
            for this_k in range(topk, max_try_k+1, topk):
                topk_external = get_candidates(query_word, this_k)
                cross = [w for w in topk_external if w in self.main_vocab]
                if cross:
                    found_words.extend(cross)
                    break

            # 2차 확장
            if not found_words and expand_level > 1:
                first_topk = get_candidates(query_word, topk)
                for w in first_topk:
                    second_topk = get_candidates(w, topk)
                    found_words.extend([ww for ww in second_topk if ww in self.main_vocab])
            found_words = list(set(found_words))

            # 철자 추천도 같이 표시
            suggestions = get_close_matches(query_word, self.external_vocab.keys(), n=5)
            return {"mode": "cross", "found_words": found_words, "suggestions": suggestions}

    def display_results(self, query, res):
        print(f"\n'{query}' 검색 결과:")
        if res.get("mode") == "main":
            print("\n📖 개인 코퍼스에서 발견:")
            for i, (word, score) in enumerate(res["main"], 1):
                print(f"  {i:2d}. {word:<15} (유사도: {score:.4f})")
        elif res.get("mode") == "cross":
            if res.get("found_words"):
                print(f"\n🌐 외부+메인 연동 후보 단어 (개인 코퍼스에도 존재):")
                print(", ".join(res["found_words"]))
            else:
                print("❌ 외부 의미망 기반 후보도 개인 문서에는 없습니다.")
            if res.get("suggestions"):
                print(f"💡 철자 유사 추천어: {', '.join(res['suggestions'])}")

    def ask_feedback(self, query, found_words):
        if not found_words: return
        print(f"\n💬 {query}와 주관적으로 가장 가까운 단어를 아래 후보 중에서 골라주세요:")
        for i, w in enumerate(found_words, 1):
            print(f"  {i:2d}. {w}")
        print("입력: 번호 또는 단어 (스킵하려면 Enter)")
        choice = input("선택: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(found_words):
                print(f"⭐️ '{query}'와 주관적으로 가장 가까운 단어: {found_words[idx-1]}")
        elif choice in found_words:
            print(f"⭐️ '{query}'와 주관적으로 가장 가까운 단어: {choice}")
        else:
            print("피드백이 저장되지 않았습니다.")

    def show_help(self):
        print("\n=== 도움말 ===")
        print("이 시스템은 개인 문서/외부 지식 의미 네트워크 기반으로")
        print("입력한 단어와 의미적으로 가까운 단어를 찾아줍니다.")
        print("개별 코퍼스 또는 외부→개인 교집합 기반 추천과 주관적 피드백을 지원합니다.")

# 실행 예시
if __name__ == "__main__":
    search_engine = MetacogAISearchUI()
    if search_engine.load_embeddings():
        search_engine.interactive_search()

