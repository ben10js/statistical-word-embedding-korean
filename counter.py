from collections import Counter

def count_word_frequencies(tokenized_sentences):
    """
    tokenized_sentences: 리스트 of 토큰 리스트 (예: [['this', 'is'], ['another', 'doc']])
    return: word_counts(Counter 객체)
    """
    word_counts = Counter()
    for sent in tokenized_sentences:
        word_counts.update(sent)
    return word_counts

def top_k_words(word_counts, k=20):
    """
    word_counts: Counter 객체
    k: 추출할 단어 개수
    return: 가장 많이 등장한 상위 k개 단어 리스트
    """
    return [w for w, _ in word_counts.most_common(k)]
