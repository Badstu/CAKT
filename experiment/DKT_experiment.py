from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import random
from data.KTData import KTData
from model.RNN import RNN_DKT
from model.CNN import CNN, CNN_3D
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

import train
import test


def init_loss_file(opt):
    # delete loss file while exist
    if os.path.exists(opt.train_loss_path):
        os.remove(opt.train_loss_path)
    if os.path.exists(opt.val_loss_path):
        os.remove(opt.val_loss_path)
    if os.path.exists(opt.test_loss_path):
        os.remove(opt.test_loss_path)


def main(**kwargs):
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.vis:
        vis = Visualizer(opt.env)
    else:
        vis = None

    init_loss_file(opt)
    if opt.data_source == "statics":
        opt.fold_dataset = True
    train_path, valid_path, test_path = init_file_path(opt)
    print(opt.fold_dataset)

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

    if opt.model_name == "CNN":
        model = CNN(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    elif opt.model_name == "CNN_3D":
        model = CNN_3D(opt.k_frames, opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    else:
        model = RNN_DKT(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)

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

    model = model.to(opt.device)

    loss_result = {}
    auc_resilt = {}
    best_test_auc = 0
    # START TRAIN
    for epoch in range(opt.max_epoch):
        if epoch < last_epoch:
            continue
        if opt.model_name == "CNN_3D":
            train_loss_meter, train_auc_meter, train_loss_list = train.train_3d(opt, vis, model, train_loader, epoch, lr,
                                                                             optimizer)
            val_loss_meter, val_auc_meter, val_loss_list = train.valid_3d(opt, vis, model, valid_loader, epoch)
            test_loss_meter, test_auc_meter, test_loss_list = test.test_3d(opt, vis, model, test_loader, epoch)
        else:
            train_loss_meter, train_auc_meter, train_loss_list = train.train(opt, vis, model, train_loader, epoch, lr, optimizer)
            val_loss_meter, val_auc_meter, val_loss_list = train.valid(opt, vis, model, valid_loader, epoch)
            test_loss_meter, test_auc_meter, test_loss_list = test.test(opt, vis, model, test_loader, epoch)

        loss_result["train_loss"] = train_loss_meter.value()[0]
        auc_resilt["train_auc"] = train_auc_meter.value()[0]
        loss_result["val_loss"] = val_loss_meter.value()[0]
        auc_resilt["val_auc"] = val_auc_meter.value()[0]
        loss_result["test_loss"] = test_loss_meter.value()[0]
        auc_resilt["test_auc"] = test_auc_meter.value()[0]

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

        best_test_auc = max(best_test_auc, test_auc_meter.value()[0], val_auc_meter.value()[0])
        print("best_test_auc is: ", best_test_auc)

        # TODO 每个epoch结束后把loss写入文件
        myutils.save_loss_file(opt, epoch, train_loss_list, val_loss_list, test_loss_list)

        # TODO 每save_every个epoch结束后保存模型参数+optimizer参数
        if epoch % opt.save_every == 0:
            myutils.save_model_weight(opt, model, optimizer, epoch, lr)

        # TODO 做lr_decay
        lr = myutils.adjust_lr(opt, optimizer, epoch, train_loss_meter.value()[0])

    # TODO 结束的时候保存final模型参数
    myutils.save_model_weight(opt, model, optimizer, epoch, lr, is_final=True)


if __name__ == '__main__':
    list_datasets = [
        ("assist2009", 110),
        ("assist2015", 100),
        ("statics", 1223),
        ("synthetic", 50)
    ]

    for data_source, output_dim in list_datasets:
        main(env="RNN_DKT",
             model_name="RNN",
             data_source=data_source,
             lr_decay=1,
             weight_decay=0,
             hidden_dim=200,
             embed_dim=200,
             input_dim=2 * output_dim,
             output_dim=output_dim,
             max_epoch=20,
             batch_size=64,
             vis=False,
             issave=False)

    # main(model_name = 'CNN_3D',
    #      k_frames=8,
    #      batch_size = 64,
    #      num_layers=2,
    #      lr=0.001,
    #      lr_decay=0.3,
    #      decay_every_epoch=5,
    #      weight_decay = 1e-5,
    #      cv_times=1,
    #      plot_every_iter=5,
    #      vis=True,
    #      issave=False)
