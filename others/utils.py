import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


## 只返回index
def qr2onehot_index(qid, rid):
    idx = int(qid)
    if int(rid) == 1:
        idx = int(qid) + 110
    return idx


def get_input_indexs(q_list, r_list, max_slen=150):
    res = []
    for idx, qid in enumerate(q_list):
        if idx == len(q_list) - 1:
            continue
        res.append(qr2onehot_index(qid, r_list[idx]))
    return np.array(res[-max_slen:])


def get_targets(q_list, r_list, max_slen=150):
    targets = []
    targets_ids = []
    targets = np.array(r_list[1:])
    qid_index = np.array(q_list[1:])
    for idx, _ in enumerate(qid_index):
        qid_index[idx] = int(qid_index[idx]) - 1
    targets_ids = np.array(qid_index)
    targets = np.array(targets[-max_slen:])
    targets_ids = np.array(targets_ids[-max_slen:])
    return targets_ids, targets


def loss_fun(y_true, y_pred):
    mask = tf.where(y_true >= 0)
    logits = tf.gather_nd(y_pred, mask)
    truth = tf.gather_nd(y_true, mask)
    return keras.losses.binary_crossentropy(truth, logits, from_logits=False)


# 精确率评价指标
def metric_precision(y_true, y_pred):
    mask = tf.where(y_true >= 0)
    y_pred = tf.gather_nd(y_pred, mask)
    y_true = tf.gather_nd(y_true, mask)
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    return precision


# 召回率评价指标
def metric_recall(y_true, y_pred):
    mask = tf.where(y_true >= 0)
    y_pred = tf.gather_nd(y_pred, mask)
    y_true = tf.gather_nd(y_true, mask)
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    recall = TP / (TP + FN)
    return recall


# F1-score评价指标
def metric_F1score(y_true, y_pred):
    mask = tf.where(y_true >= 0)
    y_pred = tf.gather_nd(y_pred, mask)
    y_true = tf.gather_nd(y_true, mask)
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2 * precision * recall / (precision + recall)
    return F1score


def prepare_data(q_list, r_list, max_slen=150):
    input_indexs = get_input_indexs(q_list, r_list, max_slen)
    target_ids, targets = get_targets(q_list, r_list, max_slen)
    input_indexs = np.array([input_indexs])
    target_ids = np.array([target_ids])
    targets = np.array([targets])
    input_indexs = keras.preprocessing.sequence.pad_sequences(input_indexs, maxlen=max_slen, dtype='int32',
                                                              padding='post', value=0)
    targets = keras.preprocessing.sequence.pad_sequences(targets, maxlen=max_slen, dtype='int32',
                                                         padding='post', value=-1)
    target_ids = keras.preprocessing.sequence.pad_sequences(target_ids, maxlen=max_slen, dtype='int32',
                                                            padding='post', value=0)
    return input_indexs, target_ids, targets


def act(state, qid, model, threshold=0.5):
    state['q_list'].append(qid)
    state['r_list'].append(0)
    x, tids, targets = prepare_data(state['q_list'], state['r_list'], 150)
    input_data = tf.data.Dataset.from_tensor_slices(({'input_x': x, 'indexs': tids, 'targets': targets}, targets))
    input_data = input_data.shuffle(buffer_size=256).batch(64).repeat()
    try:
        predict = model.predict(input_data, steps=1)
    except:
        print(1)

    if predict[-1][0] >= threshold:
        state['r_list'][-1] = 1
    return predict, state


def str2list(line):
    return str.split(line.strip('\n').lstrip(',').rstrip(','), ',')


def get_data(file):
    ## 设置参数
    sequence_len = 220
    max_slen = 150
    min_slen = 5
    ## 读取数据
    qid_list = []
    res_list = []
    list_len = 0

    with open(file, 'r') as file:
        for idx, line in enumerate(file):
            if idx % 3 == 1:
                tmp_list = str2list(line)
                list_len = len(tmp_list)
                start = 0
                while list_len > min_slen:
                    if list_len > max_slen:
                        qid_list.append(tmp_list[start:start + 150])
                        start += 150
                    else:
                        qid_list.append(tmp_list[start:])
                        start = len(tmp_list)
                    list_len = len(tmp_list) - start
            if idx % 3 == 2:
                tmp_list = str2list(line)
                list_len = len(tmp_list)
                start = 0
                while list_len > min_slen:
                    if list_len > max_slen:
                        res_list.append(tmp_list[start:start + 150])
                        start += 150
                    else:
                        res_list.append(tmp_list[start:])
                        start = len(tmp_list)
                    list_len = len(tmp_list) - start
    return qid_list, res_list
