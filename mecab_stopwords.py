from konlpy.tag import Okt

class StopwordFilter:
    def __init__(self):
        self.okt = Okt()
        # Korean POS tags to remove (Okt tagset)
        self.stop_pos = {"Josa", "Suffix", "Punctuation", "Foreign"}
        # English stopwords
        self.en_stopwords = {'the', 'and', 'of', 'to', 'for', 'is', 'in'}
        # Korean stopwords
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

# 객체생성 후 사용
# filterer = StopwordFilter()
# result = filterer.filter(sentences)