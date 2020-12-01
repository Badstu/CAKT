from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append("../")

from data.KTData import KTData
from model.EKT import EKTM_dev
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import myutils
from utils.visualize import Visualizer
from utils.config import Config
from utils.file_path import init_file_path

import run.run_ekt as run_ekt


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
    # print(opt.fold_dataset)

    # random_state = random.randint(1, 50)
    # print("random_state:", random_state)
    train_dataset = KTData(train_path, fold_dataset=opt.fold_dataset, q_numbers=opt.output_dim, opt='None')
    valid_dataset = KTData(valid_path, fold_dataset=opt.fold_dataset, q_numbers=opt.output_dim, opt='None')
    test_dataset = KTData(test_path, fold_dataset=opt.fold_dataset, q_numbers=opt.output_dim, opt='None')

    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True, collate_fn=myutils.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True, collate_fn=myutils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                             drop_last=True, collate_fn=myutils.collate_fn)

    if opt.model_name == "EKT":
        model = EKTM_dev(knowledge_length=110, knowledge_emb_size=100,
                         seq_hidden_size=100, is_text=opt.is_text, text_emb_size=100)

    random_text_token = torch.randn((100, ))

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
        if opt.model_name == "EKT":
            train_loss_meter, train_auc_meter, train_loss_list = run_ekt.train_ekt(opt, vis, model, train_loader, epoch, lr, optimizer)
            val_loss_meter, val_auc_meter, val_loss_list = run_ekt.valid_ekt(opt, vis, model, valid_loader, epoch)
            test_loss_meter, test_auc_meter, test_loss_list = val_loss_meter, val_auc_meter, val_loss_list
            # test_loss_meter, test_auc_meter, test_loss_list = run_ekt.test_ekt(opt, vis, model, test_loader, epoch)

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
                         name=k,
                         update='append')
        for k, v in auc_resilt.items():
            print("epoch:{epoch}, {k}:{v:.5f}".format(epoch=epoch, k=k, v=v))
            if opt.vis:
                vis.line(X=np.array([epoch]), Y=np.array([v]),
                         win="auc",
                         opts=dict(title="auc", showlegend=True),
                         name=k,
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
    # list_datasets = [
    #     ("assist2009", 110),
    #     ("assist2015", 100),
    #     ("statics", 1223),
    #     ("synthetic", 50)
    # ]

    main(env="EKT",
         model_name="EKT",
         data_source="assist2015",
         is_text=False,
         lr_decay=1,
         weight_decay=0,
         hidden_dim=200,
         embed_dim=200,

         input_dim=2 * 100,
         output_dim=100,
         max_epoch=5,
         batch_size=1,
         vis=False,
         issave=False)

