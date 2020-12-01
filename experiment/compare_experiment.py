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
from model.CNN_3D_mask import CNN_3D_mask
from model.Res21D import Res21D
from model.CNN_Concat import CNN_Concat
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


def run_one_setting(**kwargs):
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    print(opt.__dict__)

    if opt.vis:
        vis = Visualizer(opt.env)
    else:
        vis = None

    init_loss_file(opt)

    if opt.data_source == "statics":
        opt.fold_dataset = True
    train_path, valid_path, test_path = init_file_path(opt)
    print(opt.fold_dataset)
    train_dataset = KTData(train_path, fold_dataset=opt.fold_dataset, q_numbers=opt.output_dim, opt='None')
    test_dataset = KTData(test_path, fold_dataset=opt.fold_dataset, q_numbers=opt.output_dim, opt='None')

    print(len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                             drop_last=True, collate_fn=myutils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                             drop_last=True, collate_fn=myutils.collate_fn)


    if opt.model_name == "CNN":
        model = CNN(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    elif opt.model_name == "CNN_3D":
        model = CNN_3D(opt.k_frames, opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    elif opt.model_name == "CNN_3D_mask":
        model = CNN_3D_mask(opt.input_dim, opt.embed_dim, opt.hidden_dim,
                            opt.num_layers, opt.output_dim, opt.batch_size,
                            opt.device, max_seq_len=200)
    elif opt.model_name == "Res21D":
        model = Res21D(opt.k_frames, opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    elif opt.model_name == "CNN_Concat":
        model = CNN_Concat(opt.k_frames, opt.input_dim, opt.H, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    else:
        model = RNN_DKT(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)

    lr = opt.lr
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=opt.weight_decay,
        betas=(0.9, 0.99)
    )

    model = model.to(opt.device)

    best_test_auc = 0
    # START TRAIN
    for epoch in range(opt.max_epoch):
        if opt.model_name == "CNN_3D_mask" or opt.model_name == "Res21D" or opt.model_name == "CNN_Concat":
            torch.cuda.empty_cache()
            train_loss_meter, train_auc_meter, train_loss_list = train.train(opt, vis, model, train_loader, epoch, lr, optimizer)
            torch.cuda.empty_cache()
            test_loss_meter, test_auc_meter, test_loss_list = test.test(opt, vis, model, test_loader, epoch)

            print("epoch{}, {k}:{v:.5f}".format(epoch, k="train_auc", v=train_auc_meter.value()[0]))
            print("epoch{}, {k}:{v:.5f}".format(epoch, k="test_auc", v=test_auc_meter.value()[0]))

        best_test_auc = max(best_test_auc, test_auc_meter.value()[0])
        print("best_test_auc is: ", best_test_auc)

        # TODO 做lr_decay
        lr = myutils.adjust_lr(opt, optimizer, epoch, train_loss_meter.value()[0])

    return best_test_auc


def compare_setting():
    list_k = [1] + list(range(2, 33, 2))
    list_b = [32]
    list_layers = [1]
    list_reg = [1e-5, 1e-6, 1e-7, 0]

    list_H = [17, 19]

    list_datasets = [
        ("assist2009", 110),
        ("assist2015", 100),
        ("statics", 1223),
        ("synthetic", 50)
    ]

    list_params = []
    for H in list_H:
        for data_source, output_dim in list_datasets:
            # dataset, output_dim
            try:
                best_test_auc = run_one_setting(
                    model_name='CNN_Concat',
                    data_source=data_source,
                    k_frames=4,
                    batch_size=64,
                    num_layers=1,
                    H=H,
                    embed_dim=H*H,
                    hidden_dim=H*H,
                    input_dim=2 * output_dim,
                    output_dim=output_dim,
                    weight_decay=1e-5,
                    max_epoch=20,
                    vis=False
                    # fold_dataset=True
                )
                params = {
                    "k_frames": 4,
                    "batch_size": 64,
                    "H": H,
                    "dataset": data_source,
                    "output_dim": output_dim,
                    "best_test_auc": best_test_auc
                }
                print(params)
                list_params.append(str(params))
            except Exception as e:
                print(str(e))
                pass

    with open("checkpoints/sensitive_H.txt", "w") as f:
        f.write('\n'.join(list_params))


if __name__ == '__main__':
    compare_setting()

    # "assist2009"
    # "statics"
    # data_source = "assist2015"
    # output_dim = 100
    # k = 8
    # b = 64
    # l = 1
    #
    # best_test_auc = run_one_setting(
    #     model_name='CNN_Concat',
    #     data_source=data_source,
    #     k_frames=k,
    #     batch_size=b,
    #     num_layers=l,
    #     embed_dim=225,
    #     input_dim=2*output_dim,
    #     output_dim=output_dim,
    #     weight_decay=1e-5,
    #     max_epoch=30,
    #     vis=False
    # )
