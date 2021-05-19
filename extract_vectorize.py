import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open as custom_open
from keras.models import Model
from utils import *
import gc


temp_embedding = './datasets/embed.txt'


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post', save=False):
    """
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        if save:
            with open(temp_embedding, 'a') as f:
                print(x.tolist(), file=f)
            f.close()
        else:
            outputs.append(x)
    if not save:
        return np.array(outputs)


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """global average pooling layer, concatenated after BERT
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


def load_text(filename):
    """load input text
    return: [texts]
    """
    D = []
    with open(filename) as f:
        for l in f:
            texts = json.loads(l)[0]
            D.append(texts)
    return D


def predict(texts, tokenizer, encoder):
    """convert sentences into sentence embeddings
    """
    batch_token_ids, batch_segment_ids = [], []
    for txt in texts:
        token_ids, segment_ids = tokenizer.encode(txt, maxlen=512)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    outputs = encoder.predict([batch_token_ids, batch_segment_ids])
    return outputs


def convert(data):
    """convert all data samples
    """
    tokenizer = Tokenizer(bert_dict_path, do_lower_case=True)
    encoder = build_transformer_model(
        bert_config_path,
        bert_checkpoint_path,
    )
    output = GlobalAveragePooling1D()(encoder.output)
    encoder = Model(encoder.inputs, output)
    embeddings = []
    batch_num = len(data)
    if not os.path.exists(temp_embedding):
        for texts in tqdm(data, desc='vectorizing sentences'):
            outputs = predict(texts, tokenizer, encoder)
            embeddings.append(outputs)
        # batch_num = len(embeddings)
        sequence_padding(embeddings, save=True)
        gc.collect()
    embeddings = np.empty((batch_num, 256, 768))
    print("Converting to numpy array...")
    with custom_open(temp_embedding, 'r') as f:
        for idx, l in tqdm(enumerate(f), desc='converting to numpy array'):
            l = l.strip('\n')
            embeddings[idx] = np.array(eval(l))
    return embeddings


if __name__ == "__main__":
    data_extract_json = "./datasets/IPSQA_extract.json"
    data_extract_npy = data_extract_json[-5]
    data = load_text(data_extract_json)

    sentence_embeddings = convert(data)
    np.save(data_extract_npy, sentence_embeddings)
    print('output path: %s.npy' % data_extract_npy)
