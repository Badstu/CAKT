import numpy as np
import math
import torch
import random
from torch import nn
import utils as utils
from sklearn import metrics
import time

def train(model, params, optimizer, q_data, qa_data, a_data):
    N = int(math.floor(len(q_data) / params.batch_size))  # batch的数量

    # shuffle data
    shuffle_index = np.random.permutation(q_data.shape[0])
    q_data = q_data[shuffle_index]
    qa_data = qa_data[shuffle_index]
    a_data = a_data[shuffle_index]

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    start = time.time()

    for idx in range(N):
        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        a_batch_seq = a_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)  # 向下取整
        input_q = utils.variable(torch.LongTensor(q_one_seq), params.gpu)
        input_qa = utils.variable(torch.LongTensor(qa_batch_seq), params.gpu)
        input_a = utils.variable(torch.LongTensor(a_batch_seq), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)  # 维度换位

        model.zero_grad()
        loss, filtered_pred, filtered_target = model(input_q, input_qa, input_a, target_1d)
        loss.backward()  # 每一个batch做一次反向传播
        nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        # print("training : batch " + str(idx) + " finished!")

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)


    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    # if (idx + 1) % params.decay_epoch == 0:
    #     utils.adjust_learning_rate(optimizer, params.init_lr * params.lr_decay)
    # print('lr: ', params.init_lr / (1 + 0.75))
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    end = time.time()
    print("epoch time:" + str(end - start))

    return epoch_loss/N, accuracy, auc

def test(model, params, optimizer, q_data, qa_data, a_data):
    N = int(math.floor(len(q_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()

    for idx in range(N):

        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        a_batch_seq = a_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = utils.variable(torch.LongTensor(q_one_seq), params.gpu)
        input_qa = utils.variable(torch.LongTensor(qa_batch_seq), params.gpu)
        input_a = utils.variable(torch.LongTensor(a_batch_seq), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)

        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, input_a, target_1d)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

        # print("testing : batch " + str(idx) + " finished!")


    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc









