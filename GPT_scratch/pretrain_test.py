import MeCab

tagger = MeCab.Tagger()
print(tagger.parse("무궁화꽃이 피었습니다."))
