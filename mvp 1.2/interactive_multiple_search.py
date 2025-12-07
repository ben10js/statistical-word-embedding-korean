#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search_ui.py

통합된 검색 UI 스크립트:
- 메인/외부 임베딩 로드
- multi-metric 후보 수집(embedding, pmi, jaccard, edit, tfidf)
- OOV 브리지 탐색(외부->메인)
- 사용자 피드백 저장(feedback_map.json)
- 대화형 및 단일 쿼리 실행 지원
"""
import argparse
import json
import os
import sys
from typing import List, Tuple, Dict, Optional

import numpy as np

# similarity_utils 모듈(파일로 분리되어 있어야 합니다)
from similarity_utils import (
    l2_normalize_embeddings,
    nearest_by_word,
    nearest_by_pmi,
    nearest_by_jaccard,
    nearest_by_edit_distance,
    nearest_by_tfidf,
    aggregate_candidates,
    build_ngram_index_from_vocab,
    build_word_to_ngrams,
    save_pickle,
    load_pickle,
)

# ---------------------------
# 피드백 매핑 유틸
# ---------------------------
FEEDBACK_PATH_DEFAULT = "feedback_map.json"

def load_feedback_map(path: str = FEEDBACK_PATH_DEFAULT) -> Dict[str, str]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf8") as fr:
                return json.load(fr)
        except Exception:
            return {}
    return {}

def save_feedback_map(mapping: Dict[str, str], path: str = FEEDBACK_PATH_DEFAULT):
    with open(path, "w", encoding="utf8") as fw:
        json.dump(mapping, fw, ensure_ascii=False, indent=2)

# ---------------------------
# 브리지 탐색 (외부 코퍼스 -> 메인 코퍼스)
# ---------------------------
def find_bridge_candidates(
    query: str,
    main_vocab: Dict[str, int],
    external_vocab: Dict[str, int],
    external_emb_norm: Optional[np.ndarray],
    emb_search_func,
    initial_k: int = 20,
    max_rounds: int = 3,
    growth: int = 2,
) -> List[Tuple[str, float]]:
    """
    외부에서 query의 유사어를 찾고, 그 중 메인 코퍼스에 존재하는 단어들을 반환.
    emb_search_func: (word, emb_norm, vocab, topk) -> List[(word,score)]
    """
    found: Dict[str, float] = {}
    if query in main_vocab:
        return [(query, 1.0)]

    if external_emb_norm is None or external_vocab is None:
        return []

    k = initial_k
    tried = set()
    for round_i in range(max_rounds):
        # 1) 외부에서 직접 neighbors
        ext_neighbors = emb_search_func(query, external_emb_norm, external_vocab, topk=k)
        # filter main existence
        for w, s_ext in ext_neighbors:
            if w in main_vocab and w not in found:
                found[w] = s_ext
        if found:
            return sorted(found.items(), key=lambda x: -x[1])

        # 2) 2차 neighbors: neighbors of neighbors
        second_level = []
        for w, s_ext in ext_neighbors:
            if w in tried:
                continue
            tried.add(w)
            neigh2 = emb_search_func(w, external_emb_norm, external_vocab, topk=k)
            second_level.extend(neigh2)
        for w, s_ext in second_level:
            if w in main_vocab and w not in found:
                found[w] = s_ext
        if found:
            return sorted(found.items(), key=lambda x: -x[1])

        # increase k and continue
        k = k * growth

    return []

# ---------------------------
# interactive bridge & feedback flow
# ---------------------------
def interactive_bridge_and_feedback(ui, query: str, topk: int = 10):
    """
    ui: MetacogAISearchUI 인스턴스 (main_vocab, external_vocab, external_emb_norm 등 보유)
    사용자에게 후보 제시하고 선택시 피드백 저장.
    """
    fb_map = load_feedback_map()
    # 1) 메인 코퍼스 직접 존재 확인
    if query in ui.main_vocab:
        return [(query, 1.0)]

    # 2) 피드백 매핑 확인
    if query in fb_map:
        mapped = fb_map[query]
        print(f"[피드백 매핑 적용] '{query}' -> '{mapped}'")
        return [(mapped, 1.0)]

    # 3) 외부 브리지 탐색
    candidates = []
    if ui.external_embeddings is not None and ui.external_vocab is not None:
        candidates = find_bridge_candidates(
            query,
            ui.main_vocab,
            ui.external_vocab,
            ui.external_emb_norm,
            emb_search_func=nearest_by_word,
            initial_k=20,
            max_rounds=3,
            growth=2,
        )

    if not candidates:
        # fallback: 편집거리 기반 제안
        fuzzy = nearest_by_edit_distance(query, ui.main_vocab, topk=topk)
        print("메인 코퍼스에 연결 가능한 후보를 외부에서 찾지 못했습니다.")
        print("편집거리 기반 제안(메인 코퍼스 일부):", fuzzy)
        return []

    # 4) 후보 제시, 사용자 선택(혹은 취소)
    print("메인 코퍼스에서 발견된 후보들(외부 유사도 기반):")
    for i, (w, score) in enumerate(candidates[:topk], start=1):
        print(f" {i}. {w} (score: {score:.3f})")
    # 안전한 input 처리 (환경에 따라 EOF 발생 가능)
    try:
        ans = input("위 후보 중 가장 의도한 단어 번호를 입력하세요(없음: n, 취소: Enter): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n입력 실패(EOF). 피드백 저장 없이 후보 반환합니다.")
        return candidates

    if ans.isdigit():
        sel = int(ans)
        if 1 <= sel <= min(topk, len(candidates)):
            chosen = candidates[sel - 1][0]
            fb_map[query] = chosen
            save_feedback_map(fb_map)
            print(f"[피드백 저장] '{query}' -> '{chosen}'")
            return [(chosen, 1.0)]
    print("피드백을 저장하지 않았습니다.")
    return candidates

# ---------------------------
# MetacogAISearchUI (통합)
# ---------------------------
class MetacogAISearchUI:
    def __init__(self):
        self.main_embeddings: Optional[np.ndarray] = None
        self.main_vocab: Optional[Dict[str,int]] = None
        self.external_embeddings: Optional[np.ndarray] = None
        self.external_vocab: Optional[Dict[str,int]] = None
        self.pmi_matrix = None
        self.ngram_index = None
        self.word_to_ngrams = None
        self.tfidf_matrix = None

        self.main_emb_norm: Optional[np.ndarray] = None
        self.external_emb_norm: Optional[np.ndarray] = None

    def load_embeddings(
        self,
        main_emb_path='embeddings_ppmi_svd.npy',
        main_vocab_path='vocab_index.json',
        external_emb_path=None,
        external_vocab_path=None,
        pmi_path=None,
        tfidf_path=None,
        ngram_cache_path='ngram_cache.pkl',
    ) -> bool:
        try:
            self.main_embeddings = np.load(main_emb_path)
            with open(main_vocab_path, 'r', encoding='utf-8') as f:
                self.main_vocab = json.load(f)
            if external_emb_path and external_vocab_path:
                self.external_embeddings = np.load(external_emb_path)
                with open(external_vocab_path, 'r', encoding='utf-8') as f:
                    self.external_vocab = json.load(f)
            if pmi_path and os.path.exists(pmi_path):
                try:
                    self.pmi_matrix = np.load(pmi_path, allow_pickle=True)
                except Exception:
                    self.pmi_matrix = None
            if tfidf_path and os.path.exists(tfidf_path):
                try:
                    self.tfidf_matrix = np.load(tfidf_path, allow_pickle=True)
                except Exception:
                    self.tfidf_matrix = None

            # ngram cache load / build
            if os.path.exists(ngram_cache_path):
                try:
                    cache = load_pickle(ngram_cache_path)
                    self.ngram_index = cache.get('ngram_index')
                    self.word_to_ngrams = cache.get('word_to_ngrams')
                except Exception:
                    self.word_to_ngrams, _ = build_word_to_ngrams(self.main_vocab)
                    self.ngram_index = build_ngram_index_from_vocab(self.main_vocab)
                    save_pickle({'ngram_index': self.ngram_index, 'word_to_ngrams': self.word_to_ngrams}, ngram_cache_path)
            else:
                self.word_to_ngrams, _ = build_word_to_ngrams(self.main_vocab)
                self.ngram_index = build_ngram_index_from_vocab(self.main_vocab)
                save_pickle({'ngram_index': self.ngram_index, 'word_to_ngrams': self.word_to_ngrams}, ngram_cache_path)

            # normalize embeddings
            self.main_emb_norm = l2_normalize_embeddings(self.main_embeddings) if self.main_embeddings is not None else None
            if self.external_embeddings is not None:
                self.external_emb_norm = l2_normalize_embeddings(self.external_embeddings)
            return True
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return False
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False

    def get_all_candidates(self, query_word: str, topk: int = 10):
        # 1) main 존재 확인
        if query_word in (self.main_vocab or {}):
            # main 내 단어면 기존 루틴: embedding 등으로 후보 수집
            r = {}
            r['embedding'] = nearest_by_word(query_word, self.main_emb_norm, self.main_vocab, topk=topk*3)
            r['pmi'] = nearest_by_pmi(query_word, self.pmi_matrix, self.main_vocab, topk=topk*3) if self.pmi_matrix is not None else []
            r['jaccard'] = nearest_by_jaccard(query_word, self.ngram_index, self.main_vocab, topk=topk*10, word_to_ngrams=self.word_to_ngrams)
            r['edit_distance'] = nearest_by_edit_distance(query_word, self.main_vocab, topk=topk*3)
            r['tfidf'] = nearest_by_tfidf(query_word, self.tfidf_matrix, self.main_vocab, topk=topk*3) if self.tfidf_matrix is not None else []
            # remove self
            for m in r:
                r[m] = [(w,s) for (w,s) in r[m] if w != query_word]
            combined = aggregate_candidates(r, weights={'embedding':1.4,'pmi':0.8,'jaccard':0.25,'edit_distance':0.1,'tfidf':1.0}, topk=topk)
            return combined

        # 2) main에 없으면 피드백 맵 확인
        fb_map = load_feedback_map()
        if query_word in fb_map:
            mapped = fb_map[query_word]
            print(f"[피드백 적용] '{query_word}' -> '{mapped}'")
            return [(mapped, 1.0)]

        # 3) 브리지 탐색 (interactive_bridge_and_feedback 에서 사용자 입력 처리)
        bridged = interactive_bridge_and_feedback(self, query_word, topk=topk)
        if bridged:
            # 브리지에서 선택된 단어(또는 후보)로 최종 후보 집계 (main 기반)
            chosen_word = bridged[0][0]
            # 재사용: main에 있는 단어에 대한 기존 multi-metric 후보 수집
            return self.get_all_candidates(chosen_word, topk=topk)
        # 4) 아무 것도 못 찾으면 빈 리스트
        return []


    def display_multi_results(self, query, all_candidates, show_components: bool = True, max_metrics: int = 5):
        print(f"\n'{query}' 다중 유사도 기준 Top Candidates:")
        if not all_candidates:
            print("  (후보가 없습니다.)")
            return

        first_item = all_candidates[0]
        has_components = isinstance(first_item[1], dict) and 'components' in first_item[1]

        for i, item in enumerate(all_candidates, start=1):
            if has_components:
                word, info = item
                total_score = info.get('score', 0.0)
                comps = info.get('components', {})
                sorted_comps = sorted(comps.items(), key=lambda x: -x[1])[:max_metrics]
                print(f"{i:2d}. {word:<20} (통합: {total_score:0.4f})")
                for m_name, m_score in sorted_comps:
                    bar = self._render_bar(m_score, width=20)
                    print(f"      {m_name:12s} {m_score:0.3f} {bar}")
                if len(comps) > max_metrics:
                    other_sum = sum(v for _, v in sorted(comps.items(), key=lambda x: -x[1])[max_metrics:])
                    print(f"      {'others':12s} {other_sum:0.3f}")
            else:
                word, score = item
                print(f"{i:2d}. {word:<20} (score: {score:0.4f})")
        print("")

    def interactive_search(self):
        print("\n=== 메타인지 AI 단어 검색 시스템 (Multi-metric) ===")
        print("명령어: quit/exit/종료, help, 검색어 입력")
        while True:
            try:
                user_input = input("\n검색어를 입력하세요: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n입력 불가 또는 종료 신호. 대화형 모드를 종료합니다.")
                break

            if user_input.lower() in ['quit', 'exit', '종료']:
                print("검색을 종료합니다.")
                break
            if user_input.lower() == 'help':
                print("여러 기준(임베딩, PMI, 편집거리, TF-IDF 등)으로 유사도 종합 평가된 단어를 보여줍니다.")
                continue
            if not user_input:
                continue

            # Debug: 각 metric의 결과 출력
            print("Debug: individual metric outputs (top 10):")
            print("embedding:", nearest_by_word(user_input, self.main_emb_norm, self.main_vocab, 10))
            if self.pmi_matrix is not None:
                print("pmi:", nearest_by_pmi(user_input, self.pmi_matrix, self.main_vocab, 10))
            print("jaccard:", nearest_by_jaccard(user_input, self.ngram_index, self.main_vocab, 10, word_to_ngrams=self.word_to_ngrams))
            print("edit_distance:", nearest_by_edit_distance(user_input, self.main_vocab, 10))
            if self.tfidf_matrix is not None:
                print("tfidf:", nearest_by_tfidf(user_input, self.tfidf_matrix, self.main_vocab, 10))

            res = self.get_all_candidates(user_input, topk=10)
            self.display_multi_results(user_input, res)

# ---------------------------
# CLI entry
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="단일 쿼리 모드 (비대화형)")
    args = parser.parse_args()

    ui = MetacogAISearchUI()
    ok = ui.load_embeddings(
        main_emb_path='embeddings_ppmi_svd.npy',
        main_vocab_path='vocab_index.json',
        external_emb_path='external_embeddings_ppmi_svd.npy' if os.path.exists('external_embeddings_ppmi_svd.npy') else None,
        external_vocab_path='external_vocab_index.json' if os.path.exists('external_vocab_index.json') else None,
        pmi_path=None,
        tfidf_path=None,
        ngram_cache_path='ngram_cache.pkl'
    )
    if not ok:
        print("임베딩 로드 실패. 파일 경로를 확인하세요.")
        sys.exit(1)

    if args.query:
        q = args.query.strip()
        print(f"단일 쿼리 모드: '{q}'")
        res = ui.get_all_candidates(q, topk=10)
        ui.display_multi_results(q, res)
    else:
        ui.interactive_search()

if __name__ == "__main__":
    main()