from keybert import KeyBERT
import json
from tqdm import tqdm


def load_stopwords(path="./datasets/en_stopwords.txt"):
    with open(path, 'r') as f:
        words = f.readlines()
    return words


def keyword_extraction(doc, stopwords, topk=3):
    kw_model = KeyBERT('distilbert-base-nli-mean-tokens')
    kws = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=stopwords)
    candidates = kws[:min(len(kws), topk)]  # (kw, score)
    res = []
    for kw, _ in candidates:
        res.append(kw)
    return res


def batch_keyword_extraction(filename="./datasets/IPSQA_seq2seq.json"):
    keywords = set()
    stopwords = load_stopwords()
    with open(filename) as f:
        for l in tqdm(f, desc='extracting keywords'):
            extract_data = json.loads(l)['source_1']
            if len(extract_data) == 0:
                continue
            kw = set(keyword_extraction(extract_data, stopwords))
            keywords = keywords.union(kw)
    return keywords


def dump_keywords(filename="./datasets/IPSQA_keyword.txt"):
    keywords = batch_keyword_extraction()
    with open(filename, 'w') as f:
        for kw in keywords:
            print(kw, file=f)
    f.close()
    print("Keyword generation by keyBERT done!")


if __name__ == "__main__":
    dump_keywords()
