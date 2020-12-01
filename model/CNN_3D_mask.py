import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np


def make_layers(in_channels, out_channels):
    layers = []
    conv2d = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
    layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


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


# TODO CNN_3D_mask 把每个时刻t处理成max_seq_len长度的张量[1, max_seq_len, 15, 15]
class CNN_3D_mask(nn.Module):
    #k_frames = 8, input_dim = 220, embed_dim = 200, hidden_dim = 225, output_dim = 110
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device, max_seq_len, init_params=True):
        super(CNN_3D_mask, self).__init__()
        self.max_seq_len = max_seq_len
        self.device = device

        self.embedding = nn.Embedding(input_dim + 1, hidden_dim, padding_idx=0)

        # self.LSTM_FRONT = RNN(input_dim, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device, init_params)

        # input_size: (N, C_in, D, H, W)
        self.module_3d = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            # nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            # nn.BatchNorm3d(16),
            # nn.ReLU(inplace=True),

            # nn.Conv3d(16, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            # nn.BatchNorm3d(8),
            # nn.ReLU(inplace=True),

            nn.Conv3d(8, 4, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d(4, 1, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )  # [1 * 200 * 15 * 15] ==>> [1 * 25 * 15 * 15]

        self.module_2d = nn.ModuleList([make_layers(25, 25) for i in range(5)])

        '''
        self.module_2d = nn.Sequential(
            nn.Conv2d(25, 20, 3, 1, 1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),

            nn.Conv2d(20, 10, 3, 1, 1, bias=False),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),

            nn.Conv2d(50, 20, 3, 1, 1, bias=False), #[20 * 15 * 15]
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),

            nn.Conv2d(20, 20, 3, 2, 1, bias=False), #[20 * 8 * 8]
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),

            nn.Conv2d(20, 20, 3, 2, 1, bias=False), #[20 * 4 * 4]
            nn.BatchNorm2d(20),
            nn.Tanh()
        ) # [1 * 200 * 15 * 15] ==>> [20 * 15 * 15] ==>> [20 * 4 * 4]
        '''

        # self.LSTM_REAR = RNN(input_dim=self.k_frames * hidden_dim,
        #                      embed_dim=0,
        #                      hidden_dim=hidden_dim,
        #                      num_layers=num_layers,
        #                      output_dim=output_dim,
        #                      batch_size=batch_size,
        #                      device=device,
        #                      init_params=init_params,
        #                      embed=False)

        self.global_pooling = nn.AdaptiveAvgPool3d((1, 15, 15))

        self.fc1 = nn.Linear(2 * 225, 200, bias=True)
        self.fc2 = nn.Linear(200, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

        if init_params:
            nn.init.kaiming_normal_(self.embedding.weight)
            nn.init.kaiming_normal_(self.fc1.weight)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.kaiming_normal_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, 0)

    #x: [batch_size, max_seq_len](32, 200)
    def forward(self, x):
        self.batch_size = x.shape[0]
        self.max_seq_len = x.shape[1]

        embed_output = self.embedding(x)
        # print(embed_output.shape) #[batch, 200, 225]
        output = embed_output.contiguous().view(embed_output.shape[0], embed_output.shape[1], 15, 15)

        c3d_input = []
        for b in range(self.batch_size):
            batch_input = []
            for i in range(self.max_seq_len):
                split_output = output[b:b + 1, 0:i+1, :, :]
                if i+1 < self.max_seq_len:
                    pad = torch.zeros((1, self.max_seq_len - i - 1, 15, 15)).to(self.device)
                    split_output = torch.cat((split_output, pad), dim=1)

                batch_input.append(split_output)

            batch_input = torch.cat([x for x in batch_input], 0)
            c3d_input.append(batch_input.unsqueeze(0))

        c3d_input = torch.cat([x for x in c3d_input], 0)
        c3d_input = c3d_input.contiguous().view(self.batch_size*self.max_seq_len, self.max_seq_len, 15, 15)
        c3d_input = c3d_input.unsqueeze(1)

        output = self.module_3d(c3d_input)
        x = output.squeeze(1) # [b*200, 25, 15, 15]

        for conv_2d in self.module_2d:
            x = x + conv_2d(x)


        output = self.global_pooling(x.unsqueeze(1)) # [b*200, 1, 1, 15, 15]

        # output = self.module_2d(c2d_input)
        output = output.contiguous().view(self.batch_size*self.max_seq_len, -1) # [b*200, 225]
        output = torch.cat((output, embed_output.contiguous().view(self.batch_size*self.max_seq_len, -1)), dim=1) # [b*200, 2*225]
        # print("final_shape: ", output.shape)

        output = self.Tanh(self.fc1(output))
        output = self.sigmoid(self.fc2(output))
        return output, None

# if __name__ == '__main__':
#     m = nn.AdaptiveAvgPool3d((1, 15, 15))
#     input = torch.ones((1, 1, 25, 15, 15))
#     o = m(input)
#     print(o.shape)

#     max_seq_len = 10
#     input_dim = 220
#     embed_dim = 200
#     hidden_dim = 225
#     num_layers = 2
#     output_dim = 110
#     batch_size = 4
#     device = torch.device('cpu')
#
#     test_input = torch.randint(1, 110, (batch_size, max_seq_len))
#     model = CNN_3D_mask(input_dim, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device)
#
#     model(test_input)

