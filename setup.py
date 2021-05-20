#! -*- coding: utf-8 -*-

import streamlit as st
import tensorflow as tf
from keras.backend import set_session, get_session
import final_summary
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import text_segmentate
import jieba
import json
import numpy as np


k_sparse = 10
maxlen = 1024
data_seq2seq_json = "./datasets/IPSQA_seq2seq.json"
seq2seq_config_json = data_seq2seq_json[:-5] + '_config.json'


class AutoSummary(AutoRegressiveDecoder):
    """seq2seq decoder
    """
    def set_model(self, model):
        self.model = model

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    @AutoRegressiveDecoder.wraps(default_rtype='logits', use_states=True)
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        # 创建一个只返回最后一个token输出的新Model
        prediction = self.last_token(self.model).predict([token_ids, segment_ids])
        # states用来缓存ngram的n值
        if states is None:
            states = [0]
        elif len(states) == 1 and len(token_ids) > 1:
            states = states * len(token_ids)
        # 根据copy标签来调整概率分布
        probas = np.zeros_like(prediction[0]) - 1000  # 最终要返回的概率分布
        for i, token_ids in enumerate(inputs[0]):
            if states[i] == 0:
                prediction[1][i, 2] *= -1  # 0不能接2
            label = prediction[1][i].argmax()  # 当前label
            if label < 2:
                states[i] = label
            else:
                states[i] += 1
            if states[i] > 0:
                ngrams = self.get_ngram_set(token_ids, states[i])
                prefix = tuple(output_ids[i, 1 - states[i]:])
                if prefix in ngrams:  # 如果确实是适合的ngram
                    candidates = ngrams[prefix]
                else:  # 没有的话就退回1gram
                    ngrams = self.get_ngram_set(token_ids, 1)
                    candidates = ngrams[tuple()]
                    states[i] = 1
                candidates = list(candidates)
                probas[i, candidates] = prediction[0][i, candidates]
            else:
                probas[i] = prediction[0][i]
            idxs = probas[i].argpartition(- k_sparse)
            probas[i, idxs[:-k_sparse]] = -1000
        return probas, states

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids], topk)  # 基于beam search
        return self.tokenizer.decode(output_ids)


class SummaryModel(object):
    def __init__(self):
        self.graph = tf.get_default_graph()
        self.session = get_session()
        self.extract_model, self.seq_model = final_summary.load_model()
        self.threshold = 0.2

    def __call__(self, raw_comment):
        with self.graph.as_default():
            set_session(self.session)
            summary = self.demo_predict(raw_comment)
            return summary

    def demo_predict(self, text, topk=3):
        # preprocessing
        text = ' '.join(text.split('\n'))
        text_list = final_summary.convert.text_split(text)
        text = final_summary.preprocess(text_list)

        # model set up
        tokenizer, encoder = final_summary.load_base_model()

        # sentence vectorization
        texts = final_summary.convert.text_split(text)
        vecs = final_summary.vectorize.predict(texts, tokenizer, encoder)
        # extraction
        preds = self.extract_model.predict(vecs[None])[0, :, 0]
        preds = np.where(preds > self.threshold)[0]
        summary = ''.join([texts[i] for i in preds])
        # abstractive summary generation
        token_dict, keep_tokens, compound_tokens = json.load(
            open(seq2seq_config_json)
        )
        tk = Tokenizer(
            token_dict,
            do_lower_case=True,
            pre_tokenize=lambda s: jieba.cut(s, HMM=False)
        )
        autoSummary = AutoSummary(
            start_id=tk._token_start_id,
            end_id=tk._token_end_id,
            maxlen=maxlen // 2
        )
        autoSummary.set_tokenizer(tk)
        autoSummary.set_model(self.seq_model)
        summary = autoSummary.generate(summary, topk=topk)

        return summary


def setup_sidebar(multiselect_options):
    """sets up the sidebar elements for streamlit """

    st.sidebar.write("Please select applicable settings.")
    display_options = st.sidebar.multiselect(
        'Select which you want to display',
        multiselect_options,
        multiselect_options
    )

    return display_options


def generate_summary(raw_comment):
    summary = final_summary.demo_predict(raw_comment, streamlit=True)
    return summary


def text_split(text, limited=True, sep=None, max_len=256):
    if sep is not None:
        texts = text.split(sep)
    else:
        texts = text_segmentate(text, 1, '\n.,;?;!')
    if limited:
        texts = texts[-max_len:]
    return texts


def main():
    """main function to set up the streamlit application visuals"""
    st.set_page_config(layout="wide")
    multiselect_options = ["raw comments", "auto generated summary"]
    display_options = setup_sidebar(multiselect_options)

    # model initialization
    model = SummaryModel()

    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.beta_columns(
        (.1, 2, .2, 1, .1))

    row0_1.title("IPS Comments Summary Demo")

    with row0_2:
        st.write('')

    row0_2.subheader('A Web App by Qiyao Wu')

    row1_spacer1, row1_1, row1_spacer2 = st.beta_columns((.1, 3.2, .1))

    with row1_1:
        st.markdown(
            "Welcome to IPS Comments Summary App. This project aims to summarize IPS comments so that engineers can "
            "save some time from searching key information or answers from previous cases. Give it a go!")
        st.markdown("**To begin, please input the comment you want to summarize.** 👇")

    row2_spacer1, row2_1, row2_spacer2 = st.beta_columns((.1, 3.2, .1))
    raw_comment = ""
    with row2_1:
        user_input = st.text_input(
            "Input comment here ")
        st.markdown("**Input comment: **")
        raw_comment = user_input
        # split into sentences
        user_input = "\n".join(text_split(user_input, limited=False))
        st.text(user_input)

    if multiselect_options[1] in display_options:
        row3_spacer1, row3_1, row3_spacer3 = st.beta_columns((.1, 3.2, .1))
        with row3_1:
            st.markdown("**Summary: **")
            if len(raw_comment) > 0:
                summary = model(raw_comment)
                st.text(summary)


if __name__ == "__main__":
    main()
