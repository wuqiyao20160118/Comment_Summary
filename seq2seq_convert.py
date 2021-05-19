from extract_model import *
from bert4keras.snippets import open as custom_open


def fold_convert(data, fold):
    """convert every fold using extractive model
    """
    valid_data = data_split(data, fold, num_folds, 'valid')
    valid_x = data_split_generator_txt(data_extract_txt, fold, num_folds, 'valid', np.zeros((len(data), 256, 1)),
                                       len(valid_data) // batch_size, batch_size)
    # valid_x = data_split(data_x, fold, num_folds, 'valid')

    model.load_weights('weights/extract_model.%s.weights' % fold)
    y_pred = model.predict_generator(valid_x, steps=len(valid_data) // batch_size)
    y_pred = np.array(y_pred)[:, :, 0]

    results = []
    for d, yp in tqdm(zip(valid_data, y_pred), desc='converting'):
        yp = yp[:len(d[0])]
        yp = np.where(yp > threshold)[0]
        source_1 = ''.join([d[0][idx] for idx in yp])
        source_2 = ''.join([d[0][eval(idx)] for idx in d[1]])
        result = {
            'source_1': source_1,
            'source_2': source_2,
            'target': d[2],
        }
        results.append(result)

    return results


def convert(filename, data):
    """convert the data to seq2seq input format
    extractive summary + actual summary
    """
    seq2seq_file = custom_open(filename, 'w', encoding='utf-8')
    total_results = []
    for fold_idx in range(num_folds):
        total_results.append(fold_convert(data, fold_idx))

    # write into seq2seq file (should follow the order in the json file)
    n = 0
    while True:
        try:
            res = total_results[n % num_folds][n // num_folds]
        except:
            break
        seq2seq_file.write(json.dumps(res, ensure_ascii=False) + '\n')
        n += 1

    seq2seq_file.close()


if __name__ == "__main__":
    data = load_data(data_extract_json)
    # data_x = np.load(data_extract_npy)

    data_seq2seq_json = '_'.join(data_extract_json.split('_')[:-1]) + '_seq2seq.json'
    # convert(data_seq2seq_json, data, data_x)
    convert(data_seq2seq_json, data)
    print('output path: %s' % data_seq2seq_json)
