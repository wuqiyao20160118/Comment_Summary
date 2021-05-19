import json
import numpy as np
import re
from tqdm import tqdm
from bert4keras.snippets import open as custom_open
from bert4keras.snippets import parallel_apply
from utils import *
from bert4keras.snippets import text_segmentate

# initialization
maxLen = 256


def text_split(text, limited=True, sep=None):
    if sep is not None:
        texts = text.split(sep)
    else:
        texts = text_segmentate(text, 1, '\n.,;?;!')
    if limited:
        texts = texts[-maxLen:]
    return texts


def load_data(comment_path="./datasets/IPSQA_parsed_text", summary_path="./datasets/IPSQA_parsed_summary"):
    """load the data
    return: [(text, summary, ips_no)]
    """
    D = []
    comment_list = os.listdir(comment_path)
    for filename in comment_list:
        ips_no = os.path.splitext(filename)[0]
        comment_fn, summary_fn = os.path.join(comment_path, filename), os.path.join(summary_path, filename)
        comment_file, summary_file = custom_open(comment_fn, 'r', encoding='utf-8'), open(summary_fn, 'r', encoding='utf-8')
        comment = comment_file.read().rstrip('\n')
        comment = re.sub(r'\s+', ' ', comment)
        summary = summary_file.read().rstrip('\n')
        summary = re.sub(r'\s+', ' ', summary)
        D.append((comment, summary, ips_no))
    return D


def extract_matching(texts, summaries, start_i=0, start_j=0):
    """greedy search for pseudo extractive summary label
    algorithm：find the longest summaries sentence，then find a sentence in texts
          which is the most similar semantically. Recursively execute the function
          until all summary sentences are processed.
    """
    if len(texts) == 0 or len(summaries) == 0:
        return []
    i = np.argmax([len(s) for s in summaries])
    j = np.argmax([compute_main_metric(t, summaries[i], 'word') for t in texts])
    lm = extract_matching(texts[:j + 1], summaries[:i], start_i, start_j)
    rm = extract_matching(
        texts[j:], summaries[i + 1:], start_i + i + 1, start_j + j
    )
    return lm + [(start_i + i, start_j + j)] + rm


def extract_flow(inputs):
    """construct data flow (for parallel_apply)
    """
    text, summary, ips_no = inputs
    texts = text_split(text, True, sep=' <S_SEP> ')  # extract final maxLen sentences
    summaries = text_split(summary, False)
    print("Processing "+ips_no)
    mapping = extract_matching(texts, summaries)
    labels = sorted(set([i[1] for i in mapping]))
    pred_summary = ''.join([texts[i] for i in labels])
    labels = [str(l) for l in labels]
    metric = compute_main_metric(pred_summary, summary)
    return texts, labels, summary, str(ips_no), metric


def convert(data):
    """split into sentences, then generate the extractive summary
    """
    D = parallel_apply(
        func=extract_flow,
        iterable=tqdm(data, desc='converting the data'),
        workers=100,
        max_queue_size=200
    )
    total_metric = sum([d[4] for d in D])
    D = [d[:4] for d in D]
    print('The average metric for extractive summary is: %s' % (total_metric / len(D)))
    return D


if __name__ == "__main__":

    data = load_data()
    data = convert(data)

    data_random_order_json = "./datasets/IPSQA_random_order.json"
    data_extract_json = "./datasets/IPSQA_extract.json"

    if os.path.exists(data_random_order_json):
        idxs = json.load(custom_open(data_random_order_json))
    else:
        idxs = list(range(len(data)))
        np.random.shuffle(idxs)
        json.dump(idxs, custom_open(data_random_order_json, 'w'))

    data = [data[i] for i in idxs]

    with custom_open(data_extract_json, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print('random data index: %s' % data_random_order_json)
    print('extractive training data output path: %s' % data_extract_json)
