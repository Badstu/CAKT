import math

import numpy as np
import torch
from sklearn import metrics
from torch import nn

import utils as utils


def train(model,epoch, params, optimizer, q_a_data, q_target_data, answer_data):
    N = int(math.floor(len(q_a_data) / params.batch_size))

    # shuffle data
    # shuffle_index = np.random.permutation(q_a_data.shape[0])
    # q_a_data = q_a_data[shuffle_index]
    # q_target_data = q_target_data[shuffle_index]
    # answer_data = answer_data[shuffle_index]

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    for idx in range(N):
        q_a_seq = q_a_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        q_target_seq = q_target_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        answer_seq = answer_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (answer_seq - 1) / params.n_question
        target = np.floor(target)
        input_q_target = utils.variable(torch.LongTensor(q_target_seq), params.gpu)
        input_x = utils.variable(torch.LongTensor(q_a_seq), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)
        input_q_target_to_1d = torch.chunk(input_q_target, params.batch_size, 0)
        input_q_target_1d = torch.cat([input_q_target_to_1d[i] for i in range(params.batch_size)], 1)
        input_q_target_1d = input_q_target_1d.permute(1, 0)

        model.zero_grad()
        # print(input_x, input_x.shape)
        loss, filtered_pred, filtered_target = model(input_x, input_q_target_1d, target_1d)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)
        # print(epoch_loss / (idx+1))

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        # print(right_pred, right_target)
        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    # if (idx + 1) % params.decay_epoch == 0:
    #     utils.adjust_learning_rate(optimizer, params.init_lr * params.lr_decay)
    # print('lr: ', params.init_lr / (1 + 0.75))
    # print(all_target, all_pred)
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc


def test(model, params, optimizer,q_a_data, q_target_data, answer_data):
    N = int(math.floor(len(q_a_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()

    for idx in range(N):
        q_a_seq = q_a_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        q_target_seq = q_target_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        answer_seq = answer_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (answer_seq - 1) / params.n_question
        target = np.floor(target)
        input_q_target = utils.variable(torch.LongTensor(q_target_seq), params.gpu)
        input_x = utils.variable(torch.LongTensor(q_a_seq), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)
        input_q_target_to_1d = torch.chunk(input_q_target, params.batch_size, 0)
        input_q_target_1d = torch.cat([input_q_target_to_1d[i] for i in range(params.batch_size)], 1)
        input_q_target_1d = input_q_target_1d.permute(1, 0)

        loss, filtered_pred, filtered_target = model.forward(input_x, input_q_target_1d, target_1d)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc
