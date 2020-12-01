from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchnet import meter


def train_ekt(opt, vis, model, data_loader, epoch, lr, optimizer):
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    train_loss_list = []

    pred_labels = []
    gt_labels = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(data_loader)):
        # batch_seq: (batch_size, max_batch_len)
        # batch_label: (batch_size, max_batch_len, (next_qnumber, next_0_1))

        batch_seq = batch_seq.to(opt.device)
        each_seq = batch_seq.squeeze()
        batch_label = batch_label.float().to(opt.device)

        loss = 0
        auc = 0

        hidden_state = None
        for i, item in enumerate(each_seq):
            label = 0 if item <= 110 else 1
            label = torch.Tensor([label])
            label = label.float().to(opt.device)

            # model predict
            prediction_score, hidden_state = model(item, hidden=hidden_state)
            # print(prediction_score.item(), label.item())

            # calculate each record loss
            loss += criterion(prediction_score.view_as(label), label)

            # form predict vector to calc auc

            gt_labels.append(label.item())
            pred_labels.append(prediction_score.item())

        # if np.sum(gt_label) == len(gt_label) or np.sum(gt_label) == 0:
        #     continue

        loss /= each_seq.shape[0]
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # auc_meter.add(auc)
        loss_meter.add(loss.item())

        print(gt_label, pred_label)
        print(auc)

        train_loss_list.append(str(loss_meter.value()[0]))  # 训练到目前为止所有的loss平均值
        if opt.vis and (ii) % opt.plot_every_iter == 0:
            vis.plot("train_loss", loss_meter.value()[0])
            vis.plot("train_auc", auc_meter.value()[0])
            vis.log("epoch:{epoch}, lr:{lr:.5f}, train_loss:{loss:.5f}, train_auc:{auc:.5f}".format(epoch=epoch,
                                                                                                    lr=lr,
                                                                                                    loss=loss_meter.value()[0],
                                                                                                    auc=auc_meter.value()[0]))
    auc = roc_auc_score(gt_labels, pred_labels)
    auc_meter.add(auc)

    return loss_meter, auc_meter, train_loss_list


@torch.no_grad()
def valid_ekt(opt, vis, model, valid_loader, epoch):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    val_loss_list = []

    gt_labels = []
    pred_labels = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(valid_loader)):
        batch_seq = batch_seq.to(opt.device)
        each_seq = batch_seq.squeeze()
        batch_label = batch_label.float().to(opt.device)

        loss = 0
        auc = 0

        hidden_state = None
        for i, item in enumerate(each_seq):
            label = 0 if item <= 110 else 1
            label = torch.Tensor([label])
            label = label.float().to(opt.device)

            # model predict
            prediction_score, hidden_state = model(item, hidden=hidden_state)
            # print(prediction_score.item(), label.item())

            # calculate each record loss
            loss += criterion(prediction_score.view_as(label), label)

            # form predict vector to calc auc
            gt_labels.append(label.item())
            pred_labels.append(prediction_score.item())

        # if np.sum(gt_label) == len(gt_label) or np.sum(gt_label) == 0:
        #     continue

        loss /= each_seq.shape[0]

        # auc_meter.add(auc)
        loss_meter.add(loss.item())

        val_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值

        if opt.vis:
            vis.plot("valid_loss", loss_meter.value()[0])
            vis.plot("valid_auc", auc_meter.value()[0])

    auc = roc_auc_score(gt_labels, pred_labels)
    auc_meter.add(auc)
    with open("../checkpoints/gt_labels.csv", 'w') as f:
        f.write(",".join(gt_labels))
    with open("../checkpoints/pred_labels.csv", 'w') as f:
        f.write(",".join(pred_labels))
    if opt.vis:
        vis.log("epoch:{epoch}, valid_loss:{loss:.5f}, valid_auc:{auc:.5f}".format(epoch=epoch,
                                                                            loss=loss_meter.value()[0],
                                                                            auc=auc_meter.value()[0]))
    return loss_meter, auc_meter, val_loss_list