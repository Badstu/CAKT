from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchnet import meter

def train_ckt(opt, vis, model, data_loader, epoch, lr, optimizer):
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    train_loss_list = []
    
    all_preds = []
    all_labels = []
    for ii, (batch_len, batch_seq, batch_label, batch_past_counts, batch_repeat_gaps) in tqdm(enumerate(data_loader)):
        '''
        # input: [batch, time_step, input_size](after embedding)
        # batch_seq: (batch_size, max_batch_len)
        # batch_label: (batch_size, max_batch_len, (next_qnumber, next_0_1))
        '''
        # torch.cuda.empty_cache()
        
        max_seq_len = batch_seq.shape[1]
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)
        batch_past_counts = batch_past_counts.long().to(opt.device)
        batch_repeat_gaps = batch_repeat_gaps.long().to(opt.device)

        # TODO model apply & process output prediction
        if opt.model_name == "CKT_dev":
            # model forward propogation
            output = model(batch_seq, batch_label) # [batch_size, max_seq_len]

            next_question_number = batch_label[:, :, 0].view(-1).long()
            next_question_label = batch_label[:, :, 1]

            label = []
            mask = torch.zeros_like(output)
            for i in range(opt.batch_size):
                len = batch_len[i]
                mask[i, :len] = True
                # predict.extend(output[i, :len])
                label.extend(next_question_label[i, :len])

            predict = torch.masked_select(output, mask.bool())
            # predict = torch.Tensor(predict).to(opt.device)
            label = torch.Tensor(label).to(opt.device)
        elif opt.model_name == "CKT" or opt.model_name == "CKT_ablation":
            # model forward propogation
            output, hidden_state, _ = model(batch_seq, batch_past_counts, batch_repeat_gaps)

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
        elif opt.model_name == "CKT_CIKM":
            # model forward propogation
            output, hidden_state = model(batch_seq)

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
        ############################
        all_preds.extend(predict.tolist())
        all_labels.extend(label.tolist())

        # calculate loss
        loss = criterion(predict, label)
        loss.backward()
        loss_meter.add(loss.item())

        # update weight
        optimizer.step()
        optimizer.zero_grad()
        
        # calculate auc
        if label.sum() == label.shape[0] or label.sum() == 0:
            continue
        auc = roc_auc_score(label.cpu().data, predict.cpu().data)
        auc_meter.add(auc)

        train_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值

        if opt.vis and (ii) % opt.plot_every_iter == 0:
            vis.plot("train_loss", loss_meter.value()[0])
            vis.plot("train_auc", auc_meter.value()[0])
            vis.log("epoch:{epoch}, lr:{lr:.5f}, train_loss:{loss:.5f}, train_auc:{auc:.5f}".format(epoch = epoch,
                                                                                                lr = lr,
                                                                                                loss = loss_meter.value()[0],
                                                                                                auc = auc_meter.value()[0]))
    all_auc = roc_auc_score(all_labels, all_preds)
    return loss_meter, auc_meter, all_auc, train_loss_list

@torch.no_grad()
def valid_ckt(opt, vis, model, valid_loader, epoch):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    val_loss_list = []

    all_preds = []
    all_labels = []
    for ii, (batch_len, batch_seq, batch_label, batch_past_counts, batch_repeat_gaps) in tqdm(enumerate(valid_loader)):
        # torch.cuda.empty_cache()

        max_seq_len = batch_seq.shape[1]
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)
        batch_past_counts = batch_past_counts.long().to(opt.device)
        batch_repeat_gaps = batch_repeat_gaps.long().to(opt.device)

        # TODO model apply & process output prediction
        if opt.model_name == "CKT_dev":
            # model forward propogation
            output = model(batch_seq, batch_label) # [batch_size, max_seq_len]

            next_question_number = batch_label[:, :, 0].view(-1).long()
            next_question_label = batch_label[:, :, 1]

            label = []
            mask = torch.zeros_like(output)
            for i in range(opt.batch_size):
                len = batch_len[i]
                mask[i, :len] = True
                # predict.extend(output[i, :len])
                label.extend(next_question_label[i, :len])

            predict = torch.masked_select(output, mask.bool())
            # predict = torch.Tensor(predict).to(opt.device)
            label = torch.Tensor(label).to(opt.device)
        elif opt.model_name == "CKT" or opt.model_name == "CKT_ablation":
            # model forward propogation
            output, hidden_state, _ = model(batch_seq, batch_past_counts, batch_repeat_gaps)

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
        elif opt.model_name == "CKT_CIKM":
            # model forward propogation
            output, hidden_state = model(batch_seq)

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
        ############################
        all_preds.extend(predict.tolist())
        all_labels.extend(label.tolist())

        # calculate loss
        loss = criterion(predict, label)
        loss_meter.add(loss.item())
        
        # calculate auc
        if label.sum() == label.shape[0] or label.sum() == 0:
            continue
        auc = roc_auc_score(label.cpu().data, predict.cpu().data)
        auc_meter.add(auc)

        val_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值

        if opt.vis:
            vis.plot("valid_loss", loss_meter.value()[0])
            vis.plot("valid_auc", auc_meter.value()[0])

    all_auc = roc_auc_score(all_labels, all_preds)
    if opt.vis:
        vis.log("epoch:{epoch}, valid_loss:{loss:.5f}, valid_auc:{auc:.5f}".format(epoch=epoch,
                                                                            loss=loss_meter.value()[0],
                                                                            auc=all_auc))
    return loss_meter, auc_meter, all_auc, val_loss_list


@torch.no_grad()
def test_ckt(opt, vis, model, test_loader, epoch):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    test_loss_list = []

    all_preds = []
    all_labels = []
    for ii, (batch_len, batch_seq, batch_label, batch_past_counts, batch_repeat_gaps) in tqdm(enumerate(test_loader)):
        # torch.cuda.empty_cache()
        
        max_seq_len = batch_seq.shape[1]
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)
        batch_past_counts = batch_past_counts.long().to(opt.device)
        batch_repeat_gaps = batch_repeat_gaps.long().to(opt.device)

        # TODO model apply & process output prediction
        if opt.model_name == "CKT_dev":
            # model forward propogation
            output = model(batch_seq, batch_label) # [batch_size, max_seq_len]

            next_question_number = batch_label[:, :, 0].view(-1).long()
            next_question_label = batch_label[:, :, 1]

            label = []
            mask = torch.zeros_like(output)
            for i in range(opt.batch_size):
                len = batch_len[i]
                mask[i, :len] = True
                # predict.extend(output[i, :len])
                label.extend(next_question_label[i, :len])

            predict = torch.masked_select(output, mask.bool())
            # predict = torch.Tensor(predict).to(opt.device)
            label = torch.Tensor(label).to(opt.device)
        elif opt.model_name == "CKT" or opt.model_name == "CKT_ablation":
            # model forward propogation
            output, hidden_state, _ = model(batch_seq, batch_past_counts, batch_repeat_gaps)

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
        elif opt.model_name == "CKT_CIKM":
            # model forward propogation
            output, hidden_state = model(batch_seq)

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
        ############################
        all_preds.extend(predict.tolist())
        all_labels.extend(label.tolist())

        # calculate loss
        loss = criterion(predict, label)
        loss_meter.add(loss.item())
        
        # calculate auc
        if label.sum() == label.shape[0] or label.sum() == 0:
            continue
        auc = roc_auc_score(label.cpu().data, predict.cpu().data)
        auc_meter.add(auc)

        test_loss_list.append(str(loss_meter.value()[0])) # 到目前为止所有的loss平均值

        if opt.vis:
            vis.plot("test_loss", loss_meter.value()[0])
            vis.plot("test_auc", auc_meter.value()[0])

    all_auc = roc_auc_score(all_labels, all_preds)
    if opt.vis:
        vis.log("epoch:{epoch}, test_loss:{loss:.5f}, test_auc:{auc:.5f}".format(epoch=epoch,
                                                                            loss=loss_meter.value()[0],
                                                                            auc=all_auc))
    return loss_meter, auc_meter, all_auc, test_loss_list
