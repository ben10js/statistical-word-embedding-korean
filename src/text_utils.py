
import re
from konlpy.tag import Okt

def is_korean_or_english(token):
    return bool(re.fullmatch(r'[가-힣a-zA-Z]+', token))

def filter_korean_english_tokens(tokens):
    return [token for token in tokens if is_korean_or_english(token)]

def filter_sentences(sentences):
    filtered_sentences = []
    for sent in sentences:
        filtered_tokens = filter_korean_english_tokens(sent)
        if filtered_tokens:
            filtered_sentences.append(filtered_tokens)
    return filtered_sentences

def clean_text_basic(text):
    cleaned = re.sub(r'[^가-힣a-zA-Z\s]+', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

class StopwordFilter:
    def __init__(self):
        self.okt = Okt()
        self.stop_pos = {"Josa", "Suffix", "Punctuation", "Foreign"}
        self.en_stopwords = {'the', 'and', 'of', 'to', 'for', 'is', 'in'}
        self.ko_stopwords = {'한다', '때문', '아니', '된다', '지만', '대한', '하면서'}

    def filter(self, sentences_tokens):
        result = []
        for tokens in sentences_tokens:
            sentence = " ".join(tokens)
            toks = self.okt.pos(sentence)
            filtered = [
                word for word, pos in toks
                if pos not in self.stop_pos
                and word.lower() not in self.en_stopwords
                and word not in self.ko_stopwords
                and len(word) > 1
            ]
            result.append(filtered)
        return result
