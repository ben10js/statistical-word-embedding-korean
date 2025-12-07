# pip install spacy
# python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")

def doc_to_sent_tokens(text):
    doc = nlp(text)
    sents = []
    for sent in doc.sents:
        tokens = [t.text.lower() for t in sent if not t.is_space]
        sents.append(tokens)
    return sents

# 예: 모든 문서에서 문장 토큰 리스트 만들기
#corpus_sent_tokens = []
#for d in docs:
    #sents = doc_to_sent_tokens(d["text"])
    #corpus_sent_tokens.extend(sents)