import os
import sys
import random
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class KTData(Dataset):
    # TODO 随机划分训练测试集
    # @params opt:['train', 'valid']
    def __init__(self, csv_path, fold_dataset=False, q_numbers=110, opt='None', random_state=42):
        # knowlege: 1~110
        # label: 0/1
        self.q_numbers = q_numbers
        self.fold_dataset = fold_dataset
        if fold_dataset:
            self.min_jieduan = 5
            self.max_jieduan = 1e6
        else:
            self.min_jieduan = 10
            self.max_jieduan = 201
        self.opt = opt
        self.pn_list = []
        self.kms = []
        self.cms = []

        self.pn_list, self.kms, self.cms = self.get_data(csv_path)

        self.train_pn_list, self.valid_pn_list,\
            self.train_kms, self.valid_kms, \
            self.train_cms, self.valid_cms = train_test_split(self.pn_list, self.kms, self.cms,
                                                    test_size=0.2, random_state=random_state)

        if self.opt == 'None':
            pass
        elif self.opt == 'train':
            self.pn_list, self.kms, self.cms = self.train_pn_list, self.train_kms, self.train_cms
        elif self.opt == 'valid':
            self.pn_list, self.kms, self.cms = self.valid_pn_list, self.valid_kms, self.valid_cms

    def __len__(self):
        if self.opt == 'None':
            return len(self.pn_list)
        elif self.opt == 'train':
            return len(self.train_pn_list)
        elif self.opt == 'valid':
            return len(self.valid_pn_list)

    def __getitem__(self, idx):
        data_length = self.pn_list[idx]
        seq = self.kms[idx]
        label = self.cms[idx]
        return self.q_numbers, data_length, seq, label

    def get_data(self, csv_path):
        '''
        practice_number(pn): 15
        knowledge(km): 65,65,65,65,65,65,65,65,65,65,65,65,65,65,65
        correct(cm): 1,1,0,1,0,1,1,1,1,0,1,1,1,1,1
        '''

        pn_list = []
        km = []
        cm = []

        flag = True
        with open(csv_path) as f:
            for idx, line in enumerate(f.readlines(), 3):
                if idx % 3 == 0:
                    pn = int(line)
                    if pn < self.min_jieduan or pn > self.max_jieduan:
                        flag = False
                        continue
                    else:
                        flag = True

                    pn_list.append(pn)
                if (idx - 1) % 3 == 0 and flag:
                    line = line[:-1]
                    if line[-1] == ',':
                        line = line[:-1]
                    per_k = list(map(lambda x: int(x), line.split(',')))
                    km.append(torch.Tensor(per_k))
                if (idx - 2) % 3 == 0 and flag:
                    line = line[:-1]
                    if line[-1] == ',':
                        line = line[:-1]
                    per_c = list(map(lambda x: int(x), line.split(',')))
                    cm.append(torch.Tensor(per_c))

        # TODO 超过200的倍数，截断放到下一个， 最后小于200的，用0补到后面
        if self.fold_dataset:
            pn_list, km, cm = self.split_200(pn_list, km, cm)

        return pn_list, km, cm

    def split_200(self, pn_list, kms, cms, seq_len = 201):
        new_pn_list = []
        new_kms = []
        new_cms = []

        for i, pn in enumerate(pn_list):
            km = kms[i]
            cm = cms[i]
            len_k = len(km)
            start = 0

            # TODO 如果是小于200的，补0在后面

            while len_k >= seq_len:
                new_pn_list.append(seq_len)
                new_kms.append(km[start: start+seq_len])
                new_cms.append(cm[start: start+seq_len])
                len_k -= seq_len
                start += seq_len

            if len_k > 0:
                new_pn_list.append(len_k)
                new_kms.append(km[start:])
                new_cms.append(cm[start:])
        return new_pn_list, new_kms, new_cms

# if __name__ == '__main__':
#     file_name = "../dataset/assist2009_updated/assist2009_updated_test.csv"
#     KTData(file_name)