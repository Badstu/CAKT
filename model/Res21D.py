import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size, device):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)


    def forward(self, x):
        h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        output, hidden_state = self.LSTM(x, (h0, c0))
        return output, hidden_state


def conv3d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv2d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                     padding=(0, 1, 1), bias=False)


def conv1d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                     padding=(1, 0, 0), bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        M = int((3*3*3*in_planes*planes) / (3*3*in_planes + 3*planes))
        self.conv2d_1 = conv2d(in_planes, M)
        self.bn1_1 = nn.BatchNorm3d(M)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1d_1 = conv1d(M, planes, stride=stride)
        self.bn1_2 = nn.BatchNorm3d(planes)
        self.relu1_2 = nn.ReLU(inplace=True)

        M = int((3 * 3 * 3 * planes * planes) / (3 * 3 * planes + 3 * planes))
        self.conv2d_2 = conv2d(planes, M)
        self.bn2_1 = nn.BatchNorm3d(M)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv1d_2 = conv1d(M, planes)
        self.bn2_2 = nn.BatchNorm3d(planes)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv2d_1(x)
        out = self.bn1_1(out)
        out = self.relu1_1(out)
        out = self.conv1d_1(out)
        out = self.bn1_2(out)
        out = self.relu1_2(out)

        out = self.conv2d_2(out)
        out = self.bn2_1(out)
        out = self.relu2_1(out)
        out = self.conv1d_2(out)
        out = self.bn2_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2_2(out)
        return out


class C3DBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(C3DBlock, self).__init__()
        self.conv3d_1 = conv3d(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3d_2 = conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv3d_2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        return out


class Res21D(nn.Module):
    def __init__(self, k_frames, input_dim, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device, init_params=True):
        super(Res21D, self).__init__()
        self.k_frames = k_frames
        self.device = device
        self.inplanes = 1

        self.embedding = nn.Embedding(input_dim + 1, embed_dim, padding_idx=0)

        self.LSTM_FRONT = RNN(embed_dim, hidden_dim, num_layers, batch_size, device)

        self.layer1 = self._make_layer(C3DBlock, 4, 1)
        # self.layer2 = self._make_layer(BasicBlock, 8, 1)
        self.layer3 = self._make_layer(C3DBlock, 4, 1)
        self.layer4 = self._make_layer(C3DBlock, 1, 1)

        self.LSTM_REAR = RNN(self.k_frames * hidden_dim, 4 * hidden_dim, num_layers, batch_size, device)

        # self.global_pooling = nn.AdaptiveAvgPool3d((1, 15, 15))
        # self.fc1 = nn.Linear(hidden_dim, 200, bias=True)
        # self.fc2 = nn.Linear(200, output_dim, bias=True)
        self.fc_final = nn.Linear(4 * hidden_dim, output_dim, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride,
                          padding=0, bias=False),
                nn.BatchNorm3d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.max_seq_len = x.shape[1]
        self.clips_nums = self.max_seq_len - self.k_frames + 1
        # self.clips_nums = self.max_seq_len

        embed_output = self.embedding(x) #[batch, max_seq_len, 200]
        output, _ = self.LSTM_FRONT(embed_output) # [batch_size, max_seq_len, 225]
        output = output.contiguous().view(output.shape[0], output.shape[1], 15, 15)

        # TODO 把LSTM的输出切分成C3D的输入，每8个frame一个clip
        cnn_3d_input = []
        for b in range(self.batch_size):
            split_input = []
            for i in range(1, self.k_frames):
                tmp = output[b:b+1, :i, :, :]
                zero_padding = torch.zeros((1, self.k_frames-i, 15, 15)).to(self.device)
                tmp = torch.cat((zero_padding, tmp), 1)
                split_input.append(tmp)
            for i in range(self.clips_nums):  # time_step = 200, 200 - 8 + 1 = 193(0-192, 7-199), 193*4 = 772
                split_input.append(output[b:b + 1, i:i + self.k_frames, :, :])

            split_input = torch.cat([x for x in split_input], 0)
            cnn_3d_input.append(split_input.unsqueeze(0))

        cnn_3d_input = torch.cat([x for x in cnn_3d_input], 0)
        cnn_3d_input = cnn_3d_input.contiguous().view(cnn_3d_input.shape[0] * cnn_3d_input.shape[1], self.k_frames, 15, 15)
        cnn_3d_input = cnn_3d_input.unsqueeze(1)  # [batch_size * clip_nums, 1, k_frames, 15, 15]
        #########

        '''
        c3d_input = []
        for b in range(self.batch_size):
            batch_input = []
            for i in range(self.max_seq_len):
                split_output = output[b:b + 1, 0:i + 1, :, :]
                if i + 1 < self.max_seq_len:
                    pad = torch.zeros((1, self.max_seq_len - i - 1, 15, 15)).to(self.device)
                    split_output = torch.cat((split_output, pad), dim=1)

                batch_input.append(split_output)

            batch_input = torch.cat([x for x in batch_input], 0)
            c3d_input.append(batch_input.unsqueeze(0))

        c3d_input = torch.cat([x for x in c3d_input], 0)
        c3d_input = c3d_input.contiguous().view(self.batch_size * self.max_seq_len, self.max_seq_len, 15, 15)
        c3d_input = c3d_input.unsqueeze(1) # [16*200, 1, 200, 15, 15]
        '''

        '''
        # TODO 直接用CNN_3D的module替换掉res3D
        x = self.CNN_module_3d(cnn_3d_input)
        x = x.squeeze()

        # TODO 用LSTM_REAR代替prefix_sum
        output = output.contiguous().view(self.batch_size, self.clips_nums,
                                          -1)  # [batch_size, clips_nums, k_frames*15*15]
        output, _ = self.LSTM_REAR(output)  # [batch_size, clips_nums, 225]

        output = output.contiguous().view(self.batch_size * self.clips_nums, -1)
        output = self.sigmoid(self.fc_final(output))  # [batch_size * clips_nums, 110]
        ###############
        '''

        x = self.layer1(cnn_3d_input)
        # x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # [batch_size * clip_nums, 1, k_frames, 15, 15]
        # x = x.squeeze()

        # TODO 用LSTM_REAR代替global avg pooling
        x = x.squeeze()
        x = x.contiguous().view(self.batch_size, self.max_seq_len, -1) #[batch_size, clip_nums, 8*15*15]
        x, _ = self.LSTM_REAR(x) #[batch_size, clip_nums, 225]
        ###############

        # x = self.global_pooling(x).squeeze()

        x = x.contiguous().view(self.batch_size * self.max_seq_len, -1) # [b*clips_nums, 225]
        # x = torch.cat((x, embed_output[:, self.k_frames - 1:, :].contiguous().view(self.batch_size * self.clips_nums, -1)), dim=1)  # [b*clips_nums, 2*225]

        # x = self.Tanh(self.fc1(x))
        # output = self.sigmoid(self.fc2(x))
        output = self.fc_final(x)
        return output, None
        

# if __name__ == '__main__':
#     batch_size = 1
#     max_seq_len = 10
#     input_dim = 220
#     embed_dim = 225
#     device = torch.device('cpu')
#     model = Res21D(input_dim, embed_dim, device)
# 
#     test_input = torch.randint(1, 110, (batch_size, max_seq_len))
#     output, _ = model(test_input)
#     print(output.shape)
