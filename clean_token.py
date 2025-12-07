import re

def is_korean_or_english(token):
    """
    토큰이 한국어(한글) 또는 영어 알파벳으로만 구성되어 있는지 확인
    Args:
        token (str): 검사할 토큰
    Returns:
        bool: 한국어/영어만 있으면 True, 아니면 False
    """
    return bool(re.fullmatch(r'[가-힣a-zA-Z]+', token))

def filter_korean_english_tokens(tokens):
    """
    토큰 리스트에서 한국어/영어만 포함된 토큰들만 필터링
    Args:
        tokens (list): 토큰들의 리스트
    Returns:
        list: 필터링된 토큰 리스트
    """
    return [token for token in tokens if is_korean_or_english(token)]

def filter_sentences(sentences):
    """
    문장 리스트(토큰화된)에서 한국어/영어만 남기고 청소
    Args:
        sentences (list): 토큰화된 문장들의 리스트 [['토큰1', '토큰2'], ['토큰3', ...], ...]
    Returns:
        list: 필터링된 문장들의 리스트
    """
    filtered_sentences = []
    for sent in sentences:
        filtered_tokens = filter_korean_english_tokens(sent)
        if filtered_tokens:  # 빈 리스트가 아닌 경우만 추가
            filtered_sentences.append(filtered_tokens)
    return filtered_sentences

def clean_text_basic(text):
    """
    기본적인 텍스트 청소: 한국어/영어/공백만 남기고 나머지 제거
    Args:
        text (str): 원본 텍스트
    Returns:
        str: 청소된 텍스트
    """
    # 한국어, 영어, 공백만 남기고 나머지 제거
    cleaned = re.sub(r'[^가-힣a-zA-Z\s]+', '', text)
    # 여러 공백을 하나로 통일
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned
