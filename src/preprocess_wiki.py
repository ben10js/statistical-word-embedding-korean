def load_wiki_corpus(wiki_xml_path, max_docs=None):
    from gensim.corpora import WikiCorpus
    from tqdm import tqdm
    import re

    print(f"Loading Wikipedia corpus from {wiki_xml_path}")
    wiki = WikiCorpus(wiki_xml_path, dictionary={})
    docs = []
    for i, tokens in enumerate(tqdm(wiki.get_texts(), desc="Processing wiki articles")):
        if max_docs and i >= max_docs:
            break
        text = ' '.join(tokens)
        if len(text) > 100:
            docs.append({
                "path": f"wiki/article_{i}",
                "text": clean_wiki_text(text),
                "meta": {"title": f"article_{i}", "source": "wikipedia"}
            })
    print(f"Loaded {len(docs)} wiki documents")
    return docs

def clean_wiki_text(text):
    import re
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
