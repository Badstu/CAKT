'''
2020/09/27
this file is experiment (main) file for CAKT_WWW, the improvement version of CAKT.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.path.abspath("./"))
print(sys.path)

import time
import random
from data.KTData import KTData
from model.RNN import RNN_DKT
from model.CNN import CNN, CNN_3D
from model.CAKT import CAKT
from model.CAKT_dev import CAKT_dev
from model.CAKT_CI import CAKT_CI
from model.CAKT_ablation import CAKT_ablation
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import myutils
from utils.visualize import Visualizer
from utils.config import Config
from utils.file_path import init_file_path
from tqdm import tqdm
from torchnet import meter

from run.run_cakt import train_cakt, valid_cakt, test_cakt

def init_loss_file(opt):
    # delete loss file while exist
    if os.path.exists(opt.train_loss_path):
        os.remove(opt.train_loss_path)
    if os.path.exists(opt.val_loss_path):
        os.remove(opt.val_loss_path)
    if os.path.exists(opt.test_loss_path):
        os.remove(opt.test_loss_path)


def CAKT_main(**kwargs):
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.vis:
        vis = Visualizer(opt.env)
    else:
        vis = None

    # init_loss_file(opt)
    if opt.data_source == "statics" or opt.data_source == "assist2017":
        opt.fold_dataset = True
    train_path, valid_path, test_path = init_file_path(opt)
    print("data_source:{} fold_dataset:{}".format(opt.data_source, opt.fold_dataset))

    # random_state = random.randint(1, 50)
    # print("random_state:", random_state)
    train_dataset = KTData(train_path, fold_dataset=opt.fold_dataset, q_numbers=opt.output_dim, opt='None')
    valid_dataset = KTData(valid_path, fold_dataset=opt.fold_dataset, q_numbers=opt.output_dim, opt='None')
    test_dataset = KTData(test_path, fold_dataset=opt.fold_dataset, q_numbers=opt.output_dim, opt='None')

    # print(train_path, valid_path, test_path)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True, collate_fn=myutils.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True, collate_fn=myutils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                             drop_last=True, collate_fn=myutils.collate_fn)
    print("model name is {}, next is inital model".format(opt.model_name))
    if opt.model_name == "CAKT_dev":
        model = CAKT_dev(opt.k_frames, opt.knowledge_length, opt.concept_length, opt.knowledge_emb_size, opt.interaction_emb_size, opt.lstm_hidden_dim, opt.lstm_num_layers, opt.batch_size, opt.device)
    elif opt.model_name == "CAKT":
        model = CAKT(opt.k_frames, opt.input_dim, opt.H, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    elif opt.model_name == "CAKT_CI":
        model = CAKT_CI(opt.k_frames, opt.input_dim, opt.H, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    elif opt.model_name == "CAKT_ablation":
        print("initial abaltion model: ", opt.ablation)
        model = CAKT_ablation(opt.k_frames, opt.input_dim, opt.H, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device, opt.ablation)
    
    lr = opt.lr
    last_epoch = -1

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=opt.weight_decay,
        betas=(0.9, 0.99)
    )
    if opt.model_path:
        map_location = lambda storage, loc: storage
        checkpoint = torch.load(opt.model_path, map_location=map_location)
        model.load_state_dict(checkpoint["model"])
        last_epoch = checkpoint["epoch"]
        lr = checkpoint["lr"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    if torch.cuda.device_count() == 1:
        model = model.to(opt.device)
    elif torch.cuda.device_count() == 2:
        model = nn.DataParallel(model,device_ids=[0,1])
        model = model.to(opt.device)
    elif torch.cuda.device_count() == 4:
        model = nn.DataParallel(model,device_ids=[0,1,2,3])
        model = model.to(opt.device)

    loss_result = {}
    auc_resilt = {}
    best_test_auc = 0
    # START TRAIN
    for epoch in range(opt.max_epoch):
        torch.cuda.empty_cache()
        if epoch < last_epoch:
            continue

        if opt.model_name == "CAKT" or opt.model_name == "CAKT_CI" or opt.model_name == "CAKT_ablation":
            train_loss_meter, train_auc_meter, train_all_auc, train_loss_list = train_cakt(opt, vis, model, train_loader, epoch, lr,
                                                                                optimizer)
            val_loss_meter, val_auc_meter, val_all_auc, val_loss_list = valid_cakt(opt, vis, model, valid_loader, epoch)
            test_loss_meter, test_auc_meter, test_all_auc, test_loss_list = test_cakt(opt, vis, model, test_loader, epoch)

        loss_result["train_loss"] = train_loss_meter.value()[0]
        # auc_resilt["train_auc"] = train_auc_meter.value()[0]
        auc_resilt["train_auc"] = train_all_auc
        loss_result["val_loss"] = val_loss_meter.value()[0]
        # auc_resilt["val_auc"] = val_auc_meter.value()[0]
        auc_resilt["val_auc"] = val_all_auc
        loss_result["test_loss"] = test_loss_meter.value()[0]
        # auc_resilt["test_auc"] = test_auc_meter.value()[0]
        auc_resilt["test_auc"] = test_all_auc

        for k, v in loss_result.items():
            print("epoch:{epoch}, {k}:{v:.5f}".format(epoch=epoch, k=k, v=v))
            if opt.vis:
                vis.line(X=np.array([epoch]), Y=np.array([v]),
                         win="loss",
                         opts=dict(title="loss", showlegend=True),
                         name = k,
                         update='append')
        for k, v in auc_resilt.items():
            print("epoch:{epoch}, {k}:{v:.5f}".format(epoch=epoch, k=k, v=v))
            if opt.vis:
                vis.line(X=np.array([epoch]), Y=np.array([v]),
                         win="auc",
                         opts=dict(title="auc", showlegend=True),
                         name = k,
                         update='append')

        # best_test_auc = max(best_test_auc, test_auc_meter.value()[0], val_auc_meter.value()[0])
        best_test_auc = max(best_test_auc, test_all_auc, val_all_auc)
        print("best_test_auc is: ", best_test_auc)

        # TODO 每个epoch结束后把loss写入文件
        myutils.save_loss_file(opt, epoch, train_loss_list, val_loss_list, test_loss_list)

        # TODO 每个epoch结束后把AUC写入文件
        myutils.save_auc_file(opt, epoch, train_all_auc, val_all_auc, test_all_auc)
        
        # TODO 每save_every个epoch结束后保存模型参数+optimizer参数
        # if epoch % opt.save_every == 0:
        if best_test_auc == test_all_auc or best_test_auc == val_all_auc:
            myutils.save_model_weight(opt, model, optimizer, epoch, lr)

        # TODO 做lr_decay
        lr = myutils.adjust_lr(opt, optimizer, epoch, train_loss_meter.value()[0])

    # TODO 结束的时候保存final模型参数
    myutils.save_model_weight(opt, model, optimizer, epoch, lr, is_final=True)

    return best_test_auc



if __name__ == '__main__':

    H = 15
    knowledge_length = 110

    best_test_auc = CAKT_main(
        model_name="CAKT",
        env='CAKT',
        data_source="assist2009",
        k_frames=4,
        batch_size=80,
        num_layers=1,
        input_dim=2 * knowledge_length,
        H = H,
        embed_dim = H*H,
        output_dim=knowledge_length,

        weight_decay=1e-5,
        max_epoch=300,
        cv_times=1,
        plot_every_iter=5,

        vis=False,
        issave=False,
        issave_loss_file=True)
    print(best_test_auc)