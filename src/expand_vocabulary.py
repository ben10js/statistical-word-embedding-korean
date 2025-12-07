# expand_vocabulary.py
import numpy as np
from .embedding_utils import nearest_by_word, build_inv_vocab

def find_overlap_words(external_vocab, main_vocab):
    """
    외부 코퍼스와 메인 코퍼스에서 중복되는 단어들 찾기
    """
    return set(external_vocab.keys()) & set(main_vocab.keys())

def expand_word_pool_iterative(input_word, external_emb_norm, external_vocab, 
                              main_vocab, overlap_words, topk_start=10, max_iterations=5):
    """
    입력 단어가 메인 코퍼스에 없을 때, 외부 코퍼스를 통해 단어 풀 확장
    
    1. 외부 코퍼스에서 입력 단어의 topK 유사어 검색
    2. 그 유사어들이 메인 코퍼스에 있는지 확인
    3. 없으면 K를 늘려가며 반복
    4. 2차 확장: 1차 topK의 각각에 대해 다시 topK 검색
    """
    if input_word not in external_vocab:
        return {"status": "not_found", "candidates": []}
    
    # 1차 확장: 직접 topK 검색
    candidates = set()
    current_k = topk_start
    
    for iteration in range(max_iterations):
        try:
            neighbors = nearest_by_word(input_word, external_emb_norm, external_vocab, topk=current_k)
            
            for word, score in neighbors:
                if word in overlap_words:  # 메인 코퍼스에도 존재
                    candidates.add((word, score, "direct", iteration+1))
            
            if candidates:
                break
                
            current_k *= 2  # K를 2배씩 늘려가며 재시도
            
        except Exception as e:
            return {"status": "error", "error": str(e), "candidates": []}
    
    # 2차 확장: 1차 topK의 각각에 대해 다시 topK
    if len(candidates) < 5:  # 충분하지 않으면 2차 확장
        try:
            first_level = nearest_by_word(input_word, external_emb_norm, external_vocab, topk=10)
            
            for word1, score1 in first_level:
                try:
                    second_level = nearest_by_word(word1, external_emb_norm, external_vocab, topk=10)
                    
                    for word2, score2 in second_level:
                        if word2 in overlap_words:
                            # 2차 연결이므로 거리 페널티 적용
                            adjusted_score = score1 * score2 * 0.8  # 거리 페널티
                            candidates.add((word2, adjusted_score, "indirect", 2))
                            
                except:
                    continue
                    
        except Exception as e:
            pass  # 2차 확장 실패해도 1차 결과는 반환
    
    return {
        "status": "found" if candidates else "no_overlap",
        "input_word": input_word,
        "candidates": sorted(list(candidates), key=lambda x: x[1], reverse=True)[:20]
    }

def request_user_feedback(input_word, candidates):
    """
    사용자에게 가장 의미적으로 가까운 단어 선택 요청
    """
    print(f"\n입력 단어 '{input_word}'에 대한 후보 단어들:")
    print("다음 중에서 당신의 의도와 가장 가까운 단어를 선택해주세요:")
    
    for i, (word, score, connection_type, level) in enumerate(candidates[:10], 1):
        print(f"{i}. {word} (유사도: {score:.4f}, 연결: {connection_type})")
    
    while True:
        try:
            choice = input("\n선택 (1-10, 또는 'skip'): ").strip()
            if choice.lower() == 'skip':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < min(10, len(candidates)):
                return candidates[choice_idx]
            else:
                print("올바른 번호를 입력해주세요.")
        except ValueError:
            print("숫자를 입력해주세요.")

# 피드백 코퍼스 업데이트 함수도 추가
def update_feedback_corpus(input_word, selected_word, feedback_db):
    """
    사용자 피드백을 바탕으로 피드백 코퍼스 업데이트
    """
    if "word_connections" not in feedback_db:
        feedback_db["word_connections"] = {}
    
    feedback_db["word_connections"][input_word] = {
        "main_corpus_connection": selected_word[0],
        "similarity_score": selected_word[1],
        "connection_type": selected_word[2],
        "user_confirmed": True,
        "timestamp": "현재시간"  # 실제로는 datetime 사용
    }
    
    return feedback_db
