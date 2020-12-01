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
from model.CKT import CKT
from model.CKT_dev import CKT_dev
from model.CKT_CIKM import CKT_CIKM
from model.CKT_ablation import CKT_ablation
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

from run.run_ckt import train_ckt, valid_ckt, test_ckt

def init_loss_file(opt):
    # delete loss file while exist
    if os.path.exists(opt.train_loss_path):
        os.remove(opt.train_loss_path)
    if os.path.exists(opt.val_loss_path):
        os.remove(opt.val_loss_path)
    if os.path.exists(opt.test_loss_path):
        os.remove(opt.test_loss_path)


def CKT_main(**kwargs):
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
    if opt.model_name == "CKT_dev":
        model = CKT_dev(opt.k_frames, opt.knowledge_length, opt.concept_length, opt.knowledge_emb_size, opt.interaction_emb_size, opt.lstm_hidden_dim, opt.lstm_num_layers, opt.batch_size, opt.device)
    elif opt.model_name == "CKT":
        model = CKT(opt.k_frames, opt.input_dim, opt.H, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    elif opt.model_name == "CKT_CIKM":
        model = CKT_CIKM(opt.k_frames, opt.input_dim, opt.H, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    elif opt.model_name == "CKT_ablation":
        print("initial abaltion model: ", opt.ablation)
        model = CKT_ablation(opt.k_frames, opt.input_dim, opt.H, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device, opt.ablation)
    
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

        if opt.model_name == "CKT" or opt.model_name == "CKT_CIKM" or opt.model_name == "CKT_ablation":
            train_loss_meter, train_auc_meter, train_all_auc, train_loss_list = train_ckt(opt, vis, model, train_loader, epoch, lr,
                                                                                optimizer)
            val_loss_meter, val_auc_meter, val_all_auc, val_loss_list = valid_ckt(opt, vis, model, valid_loader, epoch)
            test_loss_meter, test_auc_meter, test_all_auc, test_loss_list = test_ckt(opt, vis, model, test_loader, epoch)

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
    
    '''
    # run CKT_dev(with memory)
    main(model_name='CKT_dev',
         env='CKT_dev',
         k_frames=2,
         batch_size=64,
         data_source='assist2009',
         knowledge_length=110,
         concept_length=20,
         knowledge_emb_size=50,
         interaction_emb_size=100,
         lstm_hidden_dim=100,
         lstm_num_layers=1,

         max_epoch=50,
         lr=0.001,
         lr_decay=0.5,
         decay_every_epoch=10,
         weight_decay=1e-5,
         cv_times=1,
         plot_every_iter=5,
         vis=True,
         issave=False)
    '''

    H = 15
    knowledge_length = 110

    best_test_auc = CKT_main(
        model_name="CKT",
        env='CKT',
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