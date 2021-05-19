import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import LayerNormalization
from bert4keras.optimizers import Adam
from bert4keras.snippets import open as custom_open
from keras.layers import *
from keras.models import Model
from utils import *


# basic configuration for extractive model
input_size = 768
hidden_size = 384
epochs = 2
batch_size = 32
threshold = 0.2
data_extract_json = "./datasets/IPSQA_extract.json"
data_extract_npy = data_extract_json[:-5] + '.npy'
data_extract_txt = './datasets/embed.txt'
data_extract_valid_txt = './datasets/embed_valid.txt'

if len(sys.argv) == 1:
    fold = 0
else:
    fold = int(sys.argv[1])


def load_data(filename):
    """load the training data
    return: [(texts, labels, summary, ips_no)]
    """
    D = []
    with custom_open(filename, encoding='utf-8') as f:
        for l in f:
            D.append(json.loads(l))
    return D


class ResidualGatedConv1D(Layer):
    """Residual Gated Convolution Module
    """
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True

    def build(self, input_shape):
        super(ResidualGatedConv1D, self).build(input_shape)
        self.conv1d = Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='same',
        )
        self.layernorm = LayerNormalization()

        if self.filters != input_shape[-1]:
            self.dense = Dense(self.filters, use_bias=False)

        self.alpha = self.add_weight(
            name='alpha', shape=[1], initializer='zeros'
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs = inputs * mask[:, :, None]

        outputs = self.conv1d(inputs)
        gate = K.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs

    def compute_output_shape(self, input_shape):
        shape = self.conv1d.compute_output_shape(input_shape)
        return shape[0], shape[1], shape[2] // 2

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        }
        base_config = super(ResidualGatedConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_model():
    x_in = Input(shape=(None, input_size))
    x = x_in

    x = Masking()(x)
    x = Dropout(0.1)(x)
    x = Dense(hidden_size, use_bias=False)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=2)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=4)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=8)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(x_in, x)
    model.compile(
        loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy']
    )
    model.summary()

    return model


# build the DGCNN model
model = build_model()


class Evaluator(keras.callbacks.Callback):
    """callback for training
    """
    def __init__(self):
        self.best_metric = 0.0

    def on_epoch_end(self, epoch, logs=None):
        valid_x = data_split_generator_txt(data_extract_valid_txt, fold, num_folds, 'valid', data_y,
                                           len(valid_data) // valid_batch_size, valid_batch_size)
        metrics = evaluate(valid_data, valid_x, threshold + 0.1)  # threshold is set to 0.3 on generation
        if metrics['main'] >= self.best_metric:  # save the best result
            self.best_metric = metrics['main']
            model.save_weights('weights/extract_model.%s.weights' % fold)
        metrics['best'] = self.best_metric
        print(metrics)


def evaluate(data_valid, data_x, threshold=0.2):
    """evaluation on validation set
    """
    y_pred = model.predict_generator(data_x, steps=len(data_valid)//valid_batch_size)
    y_pred = np.array(y_pred)[:, :, 0]
    total_metrics = {k: 0.0 for k in metric_keys}
    print("Begin evaluation........")
    for d, yp in tqdm(zip(data_valid, y_pred), desc='evaluating'):
        yp = yp[:len(d[0])]
        yp = np.where(yp > threshold)[0]
        pred_summary = ''.join([d[0][i] for i in yp])
        metrics = compute_metrics(pred_summary, d[2], 'word')
        for k, v in metrics.items():
            total_metrics[k] += v
    print("-------------Evaluation done!---------------------")
    return {k: v / len(data_valid) for k, v in total_metrics.items()}


def load_sentence_embedding(filename, batch_num):
    sentence_embedding = np.empty((batch_num, 256, 768))
    with custom_open(filename, 'r') as f:
        for idx, l in tqdm(enumerate(f), desc='converting to numpy array'):
            l = l.strip('\n')
            sentence_embedding[idx] = np.array(eval(l))
    return sentence_embedding


if __name__ == "__main__":
    # load the data
    data = load_data(data_extract_json)
    # data_x = np.load(data_extract_npy)
    # data_y = np.zeros_like(data_x[..., :1])
    data_y = np.zeros((len(data), 256, 1))
    valid_batch_size = 32

    for i, d in enumerate(data):
        for j in d[1]:
            data_y[i, eval(j)] = 1

    train_data = data_split(data, fold, num_folds, 'train')
    valid_data = data_split(data, fold, num_folds, 'valid')
    # train_x = data_split(data_x, fold, num_folds, 'train')
    # valid_x = data_split(data_x, fold, num_folds, 'valid')
    train_x = data_split_generator_txt(data_extract_txt, fold, num_folds, 'train', data_y, len(train_data) // batch_size, batch_size)
    # valid_x = data_split_generator_txt(data_extract_valid_txt, fold, num_folds, 'valid', data_y, len(valid_data) // valid_batch_size, valid_batch_size)
    # train_y = data_split(data_y, fold, num_folds, 'train')
    # valid_y = data_split(data_y, fold, num_folds, 'valid')

    # begin training
    evaluator = Evaluator()

    model.fit_generator(
        train_x,
        epochs=epochs,
        steps_per_epoch=len(train_data) // batch_size,
        callbacks=[evaluator]
    )

    # model.fit(
    #     train_x,
    #     train_y,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     callbacks=[evaluator]
    # )
else:
    model = build_model()
    model.load_weights('weights/extract_model.%s.weights' % fold)
