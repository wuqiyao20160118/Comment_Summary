import extract_convert as convert
import extract_vectorize as vectorize
import extract_model as extract
from extract_vectorize import GlobalAveragePooling1D
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from keras.models import Model
from keras.layers import Dense
import BERT_UniLM
from utils import *

if len(sys.argv) == 1:
    fold = 0
else:
    fold = int(sys.argv[1])

epochs = 50


def load_model():
    extract_model = extract.build_model()
    extract_model.load_weights('weights/extract_model.%s.weights' % 0)
    seq_model = build_transformer_model(
        bert_config_path,
        bert_checkpoint_path,
        model='bert',
        application='unilm',
        with_mlm='linear',
        keep_tokens=BERT_UniLM.keep_tokens,
        compound_tokens=BERT_UniLM.compound_tokens,
        hierarchical_position=True,
    )
    output = seq_model.get_layer('MLM-Norm').output
    # for BIO classification
    output = Dense(3, activation='softmax')(output)
    outputs = seq_model.outputs + [output]
    seq_model = Model(seq_model.inputs, outputs)
    seq_model.load_weights('weights/seq2seq_model.%s.weights' % (epochs - 1))
    return extract_model, seq_model


def predict(text, tokenizer, encoder, topk=3):
    # sentence vectorization
    texts = convert.text_split(text)
    vecs = vectorize.predict(texts, tokenizer, encoder)
    # extraction
    preds = extract.model.predict(vecs[None])[0, :, 0]
    preds = np.where(preds > extract.threshold)[0]
    summary = ''.join([texts[i] for i in preds])
    # abstractive summary generation
    summary = BERT_UniLM.autoSummary.generate(summary, topk=topk)
    # return final summary
    return summary


def save_summary(out, ips_num, summary_path="./result"):
    fn = os.path.join(summary_path, ips_num+'.txt')
    with open(fn, 'w', encoding='utf-8') as f:
        print(out, file=f)
    f.close()


def preprocess(text_list):
    filter_delete_token, filter_replace, filter_delete_line = compile_pattern()
    processed_list = []
    for txt in text_list:
        if len(txt) == 0:
            continue
        processed_list.append(post_filtering(txt, filter_delete_token, filter_replace, filter_delete_line))
    text = " ".join(processed_list)
    return text


def load_base_model():
    tokenizer = Tokenizer(bert_dict_path, do_lower_case=True)
    encoder = build_transformer_model(
        bert_config_path,
        bert_checkpoint_path,
    )
    output = GlobalAveragePooling1D()(encoder.output)
    encoder = Model(encoder.inputs, output)
    return tokenizer, encoder


def demo_predict(text, topk=3, streamlit=False):
    # preprocessing
    text = ' '.join(text.split('\n'))
    text_list = convert.text_split(text)
    text = preprocess(text_list)

    # model set up
    tokenizer, encoder = load_base_model()

    # sentence vectorization
    texts = convert.text_split(text)
    vecs = vectorize.predict(texts, tokenizer, encoder)
    # extraction
    preds = extract.model.predict(vecs[None])[0, :, 0]
    preds = np.where(preds > extract.threshold)[0]
    summary = ''.join([texts[i] for i in preds])
    # abstractive summary generation
    summary = BERT_UniLM.autoSummary.generate(summary, topk=topk)
    if streamlit:
        return summary
    print(summary)


if __name__ == '__main__':

    from tqdm import tqdm
    import json

    tokenizer = Tokenizer(bert_dict_path, do_lower_case=True)
    encoder = build_transformer_model(
        bert_config_path,
        bert_checkpoint_path,
    )
    output = GlobalAveragePooling1D()(encoder.output)
    encoder = Model(encoder.inputs, output)

    data = extract.load_data(extract.data_extract_json)
    valid_data = data_split(data, fold, num_folds, 'valid')
    total_metrics = {k: 0.0 for k in metric_keys}
    for d in tqdm(valid_data):
        text = '\n'.join(d[0])
        ips_num_str = d[-1]
        summary = predict(text, tokenizer, encoder)
        save_summary(summary, ips_num_str)
        metrics = compute_metrics(summary, d[2])
        for k, v in metrics.items():
            total_metrics[k] += v

    metrics = {k: str(v / len(valid_data)) for k, v in total_metrics.items()}
    print(metrics)

    with open("metrics.json", 'w') as metric_file:
        json.dump(metrics, metric_file)
    metric_file.close()
