from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchnet import meter

@torch.no_grad()
def test(opt, vis, model, test_loader, epoch):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    test_loss_list = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(test_loader)):
        max_seq_len = batch_seq.shape[1]
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)

        output, hidden_state = model(batch_seq)

        if opt.model_name == "CNN_3D_mask":
            # output [batch_size*200, 1]
            # TODO mask output to predict
            next_question_number = batch_label[:, :, 0].view(-1).long()
            next_question_label = batch_label[:, :, 1].view(-1)

            label = []
            mask = torch.zeros_like(output)
            for i in range(opt.batch_size):
                start = i * max_seq_len
                len = batch_len[i]
                mask[start: start+len] = True
                label.extend(next_question_label[start:start+len])

            predict = torch.masked_select(output, mask.bool())
            label = torch.Tensor(label).to(opt.device)
            if label.sum() == label.shape[0] or label.sum() == 0:
                continue
            ##############
        else:
            # TODO mask output to predict
            next_question_number = batch_label[:, :, 0].view(-1).long()
            next_question_label = batch_label[:, :, 1].view(-1)

            mask = torch.zeros_like(output)
            label = []
            for i in range(opt.batch_size):
                start = i * max_seq_len
                len = batch_len[i]
                mask[range(start, start+len), next_question_number[start:start+len] - 1] = True
                label.extend(next_question_label[start:start+len])

            predict = torch.masked_select(output, mask.bool())
            label = torch.Tensor(label).to(opt.device)
            ##############

        loss = criterion(predict, label)
        auc = roc_auc_score(label.cpu().data, predict.cpu().data)

        auc_meter.add(auc)
        loss_meter.add(loss.item())

        test_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值

        if opt.vis:
            vis.plot("test_loss", loss_meter.value()[0])
            vis.plot("test_auc", auc_meter.value()[0])

    if opt.vis:
        vis.log("epoch:{epoch}, test_loss:{loss:.5f}, test_auc:{auc:.5f}".format(epoch=epoch,
                                                                                loss=loss_meter.value()[0],
                                                                                auc=auc_meter.value()[0]))
    return loss_meter, auc_meter, test_loss_list


@torch.no_grad()
def test_3d(opt, vis, model, test_loader, epoch):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    test_loss_list = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(test_loader)):
        max_seq_len = batch_seq.shape[1]
        clips_nums = max_seq_len - opt.k_frames + 1
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)

        output, hidden_state = model(batch_seq)

        if opt.model_name == "RNN":
            # TODO 对RNN出来的结果把前k_frames个截断了
            truncat_output = []
            for b in range(opt.batch_size):
                start = b * max_seq_len + opt.k_frames - 1
                end = (b + 1) * max_seq_len
                truncat_output.append(output[start:end, :])

            output = torch.cat([x for x in truncat_output], 0)

        elif opt.model_name == "Res21D":
            # TODO mask output to predict
            next_question_number = batch_label[:, opt.k_frames-1:, 0].contiguous().view(-1).long()
            next_question_label = batch_label[:, opt.k_frames-1:, 1].contiguous().view(-1)

            mask = torch.zeros_like(output)
            label = []
            for i in range(opt.batch_size):
                start = i * clips_nums
                len = batch_len[i] - opt.k_frames + 1
                # mask[start: start+len] = True
                mask[range(start, start + len), next_question_number[start:start + len] - 1] = True
                label.extend(next_question_label[start:start + len])

            predict = torch.masked_select(output, mask.bool())
            label = torch.Tensor(label).to(opt.device)
            if label.sum() == label.shape[0] or label.sum() == 0:
                continue

        elif opt.model_name == "CNN_3D":
            # TODO mask output to predict
            next_question_number = batch_label[:, opt.k_frames-1:, 0].contiguous().view(-1).long()
            next_question_label = batch_label[:, opt.k_frames-1:, 1].contiguous().view(-1)

            mask = torch.zeros_like(output)
            label = []
            for i in range(opt.batch_size):
                start = i * clips_nums
                len = batch_len[i] - opt.k_frames + 1
                mask[range(start, start+len), next_question_number[start:start+len] - 1] = True
                label.extend(next_question_label[start:start+len])

            predict = torch.masked_select(output, mask.bool())
            label = torch.Tensor(label).to(opt.device)
            ##############

        loss = criterion(predict, label)
        auc = roc_auc_score(label.cpu().data, predict.cpu().data)

        auc_meter.add(auc)
        loss_meter.add(loss.item())

        test_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值

        if opt.vis and False:
            vis.plot("test_loss", loss_meter.value()[0])
            vis.plot("test_auc", auc_meter.value()[0])

    vis.log("epoch:{epoch}, test_loss:{loss:.5f}, test_auc:{auc:.5f}".format(epoch=epoch,
                                                                            loss=loss_meter.value()[0],
                                                                            auc=auc_meter.value()[0]))
    return loss_meter, auc_meter, test_loss_list
