import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, output_dim,
                 batch_size, device, init_params=True, embed=True):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.embed = embed

        if embed:
            self.input_dim = embed_dim
            self.embedding = nn.Embedding(input_dim + 1, embed_dim, padding_idx=0)

        self.LSTM = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        if init_params == True and embed == True:
            # nn.init.kaiming_normal_(self.fc.weight)
            # nn.init.constant_(self.fc.bias, 0)
            nn.init.kaiming_normal_(self.embedding.weight)

    def forward(self, x):
        h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        # batch_seq_pack = rnn_utils.pack_padded_sequence(x_embed, batch_len, batch_first=True)
        if self.embed:
            x = self.embedding(x)

        output, hidden_state = self.LSTM(x, (h0, c0))
        return output, hidden_state



class CNN(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device, init_params=True):
        super(CNN, self).__init__()
        self.LSTM_FRONT = RNN(input_dim, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device, init_params)

        self.module = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        ) #[4 * 15 * 15]

        self.LSTM_REAR = RNN(input_dim = 4*15*15,
                             embed_dim = 0,
                             hidden_dim = hidden_dim,
                             num_layers = num_layers,
                             output_dim=output_dim,
                             batch_size=batch_size,
                             device=device,
                             init_params=init_params,
                             embed=False)

        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmod = nn.Sigmoid()

        if init_params:
            nn.init.kaiming_normal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        lstm_f_output, _ = self.LSTM_FRONT(x)

        # TODO 把每个时刻的225维变成15*15的feature map, 作为做模型输入，输出当前时刻预测的110向量
        output = lstm_f_output.contiguous().view(lstm_f_output.shape[0] * lstm_f_output.shape[1], 15, 15)
        cnn_input = torch.unsqueeze(output, 1)

        output = self.module(cnn_input) #[batch_size * max_seq_len, 4, 11, 11]
        output = output.contiguous().view(x.shape[0], x.shape[1], -1) #[batch_size, max_seq_len, hidden_dim]

        lstm_r_output, _ = self.LSTM_REAR(output)
        output = lstm_r_output.contiguous().view(lstm_r_output.shape[0] * lstm_r_output.shape[1], -1) #[batch_size * max_seq_len, hidden_dim]

        output = self.fc(output) #[batch_size * max_seq_len, 110]

        ''' 直接传200*250的矩阵
        features = self.extract(input)
        features += input
        features = self.relu(features)

        output = self.predict(features)
        output = self.sigmod(output)
        output = output.squeeze(1)
        output = output.contiguous().view(output.shape[0] * output.shape[1], -1) #[batch_size * max_seq_len, 110]
        '''
        return output, None



# TODO 用3D conv替换LSTM_REAR
# TODO 用3D conv替换LSTM_FRONT
class CNN_3D(nn.Module):
    def __init__(self, k_frames, input_dim, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device, init_params=True):
        super(CNN_3D, self).__init__()
        self.k_frames = k_frames

        self.embedding = nn.Embedding(input_dim + 1, hidden_dim, padding_idx=0)

        self.LSTM_FRONT = RNN(input_dim, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device, init_params)

        # input_size: (N, C_in, D, H, W)
        self.module_3d = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d(4, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(1),
            nn.Tanh()
        )  # [1 * 8 * 15 * 15] ==>> [1 * 8 * 15 * 15]

        self.LSTM_REAR = RNN(input_dim=self.k_frames * hidden_dim,
                             embed_dim=0,
                             hidden_dim=hidden_dim,
                             num_layers=num_layers,
                             output_dim=output_dim,
                             batch_size=batch_size,
                             device=device,
                             init_params=init_params,
                             embed=False)

        self.fc_1 = nn.Linear(k_frames * hidden_dim, hidden_dim, bias=True)
        self.fc_final = nn.Linear(hidden_dim, output_dim, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmod = nn.Sigmoid()

        if init_params:
            nn.init.kaiming_normal_(self.embedding.weight)
            nn.init.kaiming_normal_(self.fc_final.weight)
            nn.init.constant_(self.fc_final.bias, 0)
            nn.init.kaiming_normal_(self.fc_1.weight)
            nn.init.constant_(self.fc_1.bias, 0)

    #x: [batch_size, max_seq_len](32, 200)
    def forward(self, x):
        self.batch_size = x.shape[0]
        self.max_seq_len = x.shape[1]
        self.clips_nums = self.max_seq_len - self.k_frames + 1

        # [batch_size, max_seq_len, 225]
        lstm_f_output, _ = self.LSTM_FRONT(x)
        print(self.max_seq_len, lstm_f_output.shape)
        output = lstm_f_output.contiguous().view(lstm_f_output.shape[0], lstm_f_output.shape[1], 15, 15)

        '''
        # TODO 直接用Embbeding的向量作为C3D的输入，得到225维的embbeding
        embed_output = self.embedding(x)
        output = embed_output.contiguous().view(embed_output.shape[0], embed_output.shape[1], 15, 15)
        # print(x.shape, embed_output.shape, output.shape)
        '''

        # TODO 把LSTM的输出切分成C3D的输入，每8个frame一个clip
        cnn_3d_input = []
        for b in range(self.batch_size):
            split_input = []
            for i in range(self.clips_nums): # time_step = 200, 200 - 8 + 1 = 193(0-192, 7-199), 193*4 = 772
                split_input.append(output[b:b+1, i:i+self.k_frames, :, :])

            split_input = torch.cat([x for x in split_input], 0)
            cnn_3d_input.append(split_input.unsqueeze(0))

        cnn_3d_input = torch.cat([x for x in cnn_3d_input], 0)
        cnn_3d_input = cnn_3d_input.contiguous().view(cnn_3d_input.shape[0] * cnn_3d_input.shape[1], self.k_frames, 15, 15)
        cnn_3d_input = cnn_3d_input.unsqueeze(1) #in_channel = 1
        #########

        output = self.module_3d(cnn_3d_input) #[batch_size * (max_seq_len - 8 + 1), 1, k_frames, 15, 15]
        output = output.squeeze()

        '''
        # TODO prefix_sum -> avg
        output = output.contiguous().view(self.batch_size, self.clips_nums, -1)

        prefix_sum = torch.zeros_like(output)
        for b in range(self.batch_size):
            prefix_sum[b, 0, :] = output[b, 0, :]
            for i in range(1, self.clips_nums):
                prefix_sum[b, i, :] = output[b, i, :] + prefix_sum[b, i - 1, :] * i
                prefix_sum[b, i, :] = prefix_sum[b, i, :] / (i + 1)
                # output[b, i, :] = output[b, i, :] + output[b, i - 1, :] * i
                # output[b, i, :] = output[b, i, :] / (i + 1)

        output = prefix_sum.contiguous().view(self.batch_size * self.clips_nums, -1) #[batch_size * clips_nums, k_frames*15*15]

        output = self.fc_1(output)
        output = self.fc_final(output)
        ##########
        '''

        # TODO 用LSTM_REAR代替prefix_sum
        output = output.contiguous().view(self.batch_size, self.clips_nums, -1) #[batch_size, clips_nums, k_frames*15*15]
        output, _ = self.LSTM_REAR(output) #[batch_size, clips_nums, 225]

        output = output.contiguous().view(self.batch_size * self.clips_nums, -1)
        output = self.fc_final(output) #[batch_size * clips_nums, 110]
        return output, None
