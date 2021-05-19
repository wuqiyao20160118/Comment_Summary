import numpy as np
from rouge import Rouge
import os
import sys
from bert4keras.snippets import open as custom_open
import jieba
from collections import defaultdict
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import lda
import re


sys.setrecursionlimit(1000000)

num_folds = 15

user_dict_path = './datasets/user_dict.txt'
jieba.load_userdict(user_dict_path)
jieba.initialize()

data_json = './datasets'

if not os.path.exists('weights'):
    os.mkdir('weights')

# bert configuration
# bert_config_path = '/home/allen/data2/qiyaowu/data/bert/bert_uncased_L-12_H-768_A-12/config.json'
# bert_checkpoint_path = '/home/allen/data2/qiyaowu/data/bert/bert_uncased_L-12_H-768_A-12/bert_model.ckpt'
# bert_dict_path = '/home/allen/data2/qiyaowu/data/bert/bert_uncased_L-12_H-768_A-12/vocab.txt'
bert_config_path = 'C:\\Users\\qiyaowu\\Documents\\Python_Project\\bert_checkpoint/bert_uncased_L-12_H-768_A-12/config.json'
bert_checkpoint_path = 'C:\\Users\\qiyaowu\\Documents\\Python_Project\\bert_checkpoint/bert_uncased_L-12_H-768_A-12/bert_model.ckpt'
bert_dict_path = 'C:\\Users\\qiyaowu\\Documents\\Python_Project\\bert_checkpoint/bert_uncased_L-12_H-768_A-12/vocab.txt'

metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']
rouge = Rouge()


def is_string(s):
    return isinstance(s, str)


def load_user_dict(filename):
    """load user customized dictionary
    """
    user_dict = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            w = l.split()[0]
            user_dict.append(w)
    return user_dict


def data_split(data, fold, num_folds, mode):
    """
    split training set and validation set
    """
    if mode == 'train':
        D = [d for i, d in enumerate(data) if i % num_folds != fold]
    else:
        D = [d for i, d in enumerate(data) if i % num_folds == fold]

    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D


def data_split_generator_txt(filename, fold, num_folds, mode, target, total_step, batch_size=32):
    count = 0
    step = 0
    D, T = [], []
    if mode == 'train':
        while True:
            with custom_open(filename, 'r') as train_file:
                print("Start reading the training input from the beginning............")
                for i, d in enumerate(train_file):
                    if i % num_folds != fold:
                        d = d.strip('\n')
                        d = d.strip()
                        if count == 0:
                            D, T = [], []
                        D.append(eval(d))
                        T.append(target[i])
                        count += 1
                        if count % batch_size == 0:
                            count = 0
                            yield np.array(D), np.array(T)
                            step += 1
                            if step == total_step:
                                step = 0
                                break
            train_file.close()
    else:
        while True:
            with custom_open(filename, 'r') as valid_file:
                print("Start reading the validation input from the beginning............")
                for i, d in enumerate(valid_file):
                    if i % num_folds == fold:
                        d = d.strip('\n')
                        d = d.strip()
                        if count == 0:
                            D, T = [], []
                        D.append(eval(d))
                        T.append(target[i])
                        count += 1
                        if count % batch_size == 0:
                            count = 0
                            yield np.array(D), np.array(T)
                            step += 1
            valid_file.close()


def longest_common_subsequence(source, target):
    """最长公共子序列（source和target的最长非连续子序列）
    返回：子序列长度, 映射关系（映射对组成的list）
    注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    """
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]


def compute_rouge(source, target, unit='word'):
    """
    compute rouge-1、rouge-2、rouge-l
    """
    if unit == 'word':
        source = jieba.cut(source, HMM=False)
        target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_metrics(source, target, unit='word'):
    """
    compute all metrics
    """
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics


def compute_main_metric(source, target, unit='word'):
    """
    compute the main metric
    """
    return compute_metrics(source, target, unit)['main']


def lda_extraction(path="./datasets/IPSQA_parsed_summary", out_path="./datasets/keyword_lda.txt"):
    fnames = os.listdir(path)
    text = []
    ignore = set(stopwords.words('english'))
    stemmer = WordNetLemmatizer()
    for fn in fnames:
        with open(os.path.join(path, fn), 'r', encoding='utf-8') as f:
            paragraph = f.readlines()[0]
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                words = word_tokenize(sentence)
                stemmed = []
                for word in words:
                    if word not in ignore:
                        stemmed.append(stemmer.lemmatize(word))
                text.append(' '.join(stemmed))
        f.close()

    vec = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    ngrams = vec.fit_transform(text)
    vocab = vec.get_feature_names()

    model = lda.LDA(n_topics=int(len(text) * 0.1), random_state=1)
    model.fit(ngrams)

    n_top_words = 1
    topic_word = model.topic_word_

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        with open(out_path, 'a', encoding='utf-8') as f:
            print(' '.join(topic_words).strip('\n'), file=f)
        f.close()
    print('{} has been saved!'.format(out_path))


def compile_pattern():
    # regex filter set
    filter_delete_token = [
        r'\[.*\/.*\/.*\]',
        r'Note originally created by [\w!#$%&\'*+/=?^_`{|}~-]+(?:\.[\w!#$%&\'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?',
        r'Note originally created by',
        r'\[.*\..*\..*\]',
        r':+\s+\.',
        r'Communication Log:.*:*.*:',
        r'\(message.*\/.*\)',
        r'[\*]+.*Reply from:.*[\*]+',
        r'.*\/\s+\/\s+:',
        r'Environmental Details',
        r'Example of Error',
        r'Error Msg\s*:',
        r'^\[.*\]',
        r'^\s*[:|?]',
        r'^\s*\**'
    ]
    filter_replace = [
        r'[\w!#$%&\'*+/=?^_`{|}~-]+(?:\.[\w!#$%&\'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?',
    ]
    filter_delete_line = [
        r'Original Case Owner:',
        r'Under Investigation',
        r'PAE use only',
        r'IP Product',
        r'IP category',
        r'Suspected Platform Area',
        r'SUB AREA',
        r'ALTERA SOFTWARE',
        r'MDK Version',
        r'Communication Owner',
        r'Military/Aerospace/Government',
        r'Actual Date Closed',
        r'Maintain Question',
        r'Executive Summary',
        r'IPS Request Type',
        r'Related Platform',
        r'Memory config',
        r'Domain of issue relation',
        r'email trail for issue',
        r'email consultation on CvP'
    ]
    res_filter_delete_token, res_filter_replace, res_filter_delete_line = [], [], []
    for pattern in filter_delete_token:
        res_filter_delete_token.append(re.compile(pattern))
    for pattern in filter_replace:
        res_filter_replace.append(re.compile(pattern))
    for pattern in filter_delete_line:
        res_filter_delete_line.append(re.compile(pattern))
    return tuple(res_filter_delete_token), tuple(res_filter_replace), tuple(res_filter_delete_line)


def post_filtering(text, filter_delete_token=(), filter_replace=(), filter_delete_line=()):
    for pattern in filter_delete_token:
        text = re.sub(pattern, "", text)
    for pattern in filter_replace:
        text = re.sub(pattern, "[UNK]", text)
    for pattern in filter_delete_line:
        if re.search(pattern, text):
            text = ""
            break
    text = text.strip()
    return text


if __name__ == "__main__":
    lda_extraction()
