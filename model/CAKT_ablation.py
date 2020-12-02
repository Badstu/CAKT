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
        if torch.cuda.device_count() > 1:
            self.batch_size = batch_size // torch.cuda.device_count()
        else:
            self.batch_size = batch_size
        self.device = device

        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        
        if torch.cuda.device_count() > 1:
            self.LSTM.flatten_parameters()

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
        M = int((3 * 3 * 3 * in_planes * planes) / (3 * 3 * in_planes + 3 * planes))
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

# TODO 实现SE BLOCK
class SEBlock_C3D(nn.Module):
    def __init__(self, k_frames, reduction=2):
        super(SEBlock_C3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((k_frames, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(k_frames, k_frames // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(k_frames // reduction, k_frames, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        @ x: [batch_size, channel, k_frames, 20, 100]
        应该实现在residual之前, 在k_frames的维度上做SE
        https://blog.csdn.net/bl128ve900/article/details/93778729
        '''
        b, c, k, _, _ = x.shape
        y = self.avg_pool(x).view(b, c, k) # [b, c, k_frames]
        y = self.fc(y).view(b, c, k, 1, 1)
        return x * y.expand_as(x) # [batch_size, channel, k_frames, 20, 100]

class C3DBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None, k_frames=8):
        super(C3DBlock, self).__init__()
        self.conv3d_1 = conv3d(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3d_2 = conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.se_layer = SEBlock_C3D(k_frames, reduction=2)

    def forward(self, x):
        residual = x

        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv3d_2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_layer(out)
        out += residual
        out = self.relu2(out)
        return out


class CCAFusion(nn.Module):
    def __init__(self):
        super(CCAFusion, self).__init__()
    
    def forward(self, c3d_feature, lstm_feature):
        pass


class CAKT_ablation(nn.Module):
    def __init__(self, k_frames, input_dim, H, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device,
                 ablation="None", init_params=True):
        super(CAKT_ablation, self).__init__()
        self.k_frames = k_frames
        self.know_length = output_dim
        self.device = device
        self.ablation = ablation
        self.inplanes = 1
        self.H = H
        self.S = nn.Parameter(torch.ones(1))
        self.pc_num = 200 # past_count
        self.pc_embed_dim = 10 # past_count_embed_dim
        self.rg_num = 201 # repeat_gap
        self.rg_embed_dim = 10

        # pc_embed 和 x_embed
        self.pc_embedding = nn.Embedding(self.pc_num + 1, self.pc_embed_dim, padding_idx=0)
        self.rg_embedding = nn.Embedding(self.rg_num + 1, self.rg_embed_dim, padding_idx=0)
        self.embedding = nn.Embedding(input_dim + 1, embed_dim, padding_idx=0)
        
        # 直接把forgetting特征拼接到embed output上
        self.LSTM_FRONT = RNN(embed_dim, hidden_dim, num_layers, batch_size, device)

        self.layer1 = self._make_layer(C3DBlock, 4, 1)
        self.layer2 = self._make_layer(C3DBlock, 8, 1)
        self.layer3 = self._make_layer(C3DBlock, 4, 1)
        self.layer4 = self._make_layer(C3DBlock, 1, 1)

        self.LSTM_RECENT = RNN(hidden_dim, hidden_dim, 1, batch_size, device)

        self.LSTM_REAR = RNN(hidden_dim, hidden_dim, num_layers, batch_size, device)

        self.global_pooling = nn.AdaptiveAvgPool3d((1, self.H, self.H))
        self.fc_pooling = nn.Linear(self.k_frames * hidden_dim, hidden_dim, bias=True)

        # self.fc_gate = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.fc_c3d_gate = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.fc_lstm_gate = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.fc_final = nn.Linear(hidden_dim, output_dim, bias=True)

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
        layers.append(block(self.inplanes, planes, stride, downsample, k_frames=self.k_frames))
        self.inplanes = planes

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, past_counts, repeat_gaps):
        '''
        x: [batch_size, max_seq_len]
        past_counts: same shape as x, past trail counts of each element
        repeat_gaps: same shape as x, recent repeate time gaps of each element with log
        detail description can be found in "utils/extract_forgetting_feature.py"
        '''
        self.batch_size = x.shape[0]
        self.max_seq_len = x.shape[1]
        self.clips_nums = self.max_seq_len - self.k_frames + 1
        # self.clips_nums = self.max_seq_len

        # TODO 对x进行embedding
        embed_output = self.embedding(x)  # [batch_size, max_seq_len, 205] embed_dim=205

        input_features = embed_output

        # TODO 将embedding直接输入到LSTM上
        lstm_output, _ = self.LSTM_FRONT(input_features)  # [batch_size, max_seq_len, 225]

        '''
        # TODO 把embedding的输出切分成C3D的输入，每k个frame一个clip
        # output: [batch_size, max_seq_len, 15, 15]
        output = input_features.contiguous().view(input_features.shape[0], input_features.shape[1], self.H, self.H)
        cnn_3d_input = []
        for b in range(self.batch_size):
            split_input = []
            for i in range(1, self.k_frames):
                tmp = output[b:b + 1, :i, :, :]
                zero_padding = torch.zeros((1, self.k_frames - i, self.H, self.H)).to(self.device)
                tmp = torch.cat((zero_padding, tmp), 1)
                split_input.append(tmp)
            for i in range(self.clips_nums):  # time_step = 200, 200 - 8 + 1 = 193(0-192, 7-199), 193*4 = 772
                split_input.append(output[b:b + 1, i:i + self.k_frames, :, :])

            split_input = torch.cat([x for x in split_input], 0)
            cnn_3d_input.append(split_input.unsqueeze(0))

        # cnn_3d_input: [batch_size, max_seq_len, k_frames, H, H]
        cnn_3d_input = torch.cat([x for x in cnn_3d_input], 0)
        ###############
        '''

        # TODO 用历史相同题目来拼C3D的输入，并根据距离当前题目的时间过softmax，给每个frame一个权重
        # output: [batch_size, max_seq_len, 15, 15]
        output = input_features.contiguous().view(input_features.shape[0], input_features.shape[1], self.H, self.H)
        cnn_3d_input = []
        for b in range(self.batch_size):
            record = x[b:b+1].squeeze()
            record_feature = output[b:b+1].squeeze()
            map_q_frame = {}
    
            tmp_input = []
            for idx in range(self.max_seq_len):
                value = (record[idx].item() - 1) % self.know_length + 1 # 110 -> 110
                feature = record_feature[idx]
                # formulate dict
                if value in map_q_frame:
                    map_q_frame[value].append(idx)
                else:
                    map_q_frame[value] = [idx]
            
                relate_frame_index = map_q_frame[value]

                if self.ablation == "NO_EXP_DECAY": 
                    if len(relate_frame_index) < self.k_frames:
                        rf_idx = torch.Tensor(relate_frame_index).long().to(self.device)
                        tmp = record_feature.index_select(0, rf_idx)
                        tmp = tmp
                        # zero pad
                        zero_padding = torch.zeros((self.k_frames - rf_idx.shape[0], self.H, self.H)).to(self.device)
                        tmp = torch.cat([zero_padding, tmp], dim=0)
                        tmp_input.append(tmp.unsqueeze(0))
                    else:
                        rf_idx = torch.Tensor(relate_frame_index[-self.k_frames:]).long().to(self.device)
                        tmp = record_feature.index_select(0, rf_idx)
                        tmp = tmp
                        tmp_input.append(tmp.unsqueeze(0))
                else:
                    if len(relate_frame_index) < self.k_frames:
                        rf_idx = torch.Tensor(relate_frame_index).long().to(self.device)
                        tmp = record_feature.index_select(0, rf_idx)
                        # time interval exponential decay | exp(-\delta t / S)
                        time_interval = rf_idx - idx
                        time_interval = torch.exp(time_interval * self.S).unsqueeze(1).unsqueeze(2)
                        tmp = tmp * time_interval
                        # zero pad
                        zero_padding = torch.zeros((self.k_frames - rf_idx.shape[0], self.H, self.H)).to(self.device)
                        tmp = torch.cat([zero_padding, tmp], dim=0)
                        tmp_input.append(tmp.unsqueeze(0))
                    else:
                        rf_idx = torch.Tensor(relate_frame_index[-self.k_frames:]).long().to(self.device)
                        tmp = record_feature.index_select(0, rf_idx)
                        # time interval exponential decay | exp(-\delta t / S)
                        time_interval = rf_idx - idx
                        time_interval = torch.exp(time_interval * self.S).unsqueeze(1).unsqueeze(2)
                        tmp = tmp * time_interval
                        tmp_input.append(tmp.unsqueeze(0))
        
            tmp_input = torch.cat([x for x in tmp_input], 0)
            cnn_3d_input.append(tmp_input.unsqueeze(0))

        # cnn_3d_input: [batch_size, max_seq_len, k_frames, H, H]
        cnn_3d_input = torch.cat([x for x in cnn_3d_input], 0)
        ###############

        if self.ablation == "LSTM_RECENT":
            # TODO ablation: 用LSTM代替C3D模块（只处理最近k个frames）
            output = []
            for i in range(self.max_seq_len):
                lstm_recent_input = cnn_3d_input[:, i, :, :, :].contiguous().view(self.batch_size, self.k_frames, -1)
                _, (h, c) = self.LSTM_RECENT(lstm_recent_input)
                output.append(h)
            x = torch.cat([x for x in output], dim=0).transpose(0, 1) # [batch_size, max_seq_len, 225]
            ###############
        else:
            # TODO C3D模块
            cnn_3d_input = cnn_3d_input.contiguous().view(cnn_3d_input.shape[0] * cnn_3d_input.shape[1], self.k_frames, self.H, self.H)
            cnn_3d_input = cnn_3d_input.unsqueeze(1)  # [batch_size * clip_nums, 1, k_frames, 15, 15]
            x = self.layer1(cnn_3d_input)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)  # [batch_size * max_seq_len, 1, k_frames, 15, 15]

            if self.ablation == "FC_POOLING":
                # TODO ablation: 用FC代替global pooling
                x = self.fc_pooling(x.view(x.shape[0], -1)) # [batch_size * max_seq_len, 225]
                x = x.contiguous().view(self.batch_size, self.max_seq_len, -1)  # [batch_size, max_seq_len, 225]
            else:
                # TODO global pooling
                x = self.global_pooling(x) # [batch_size * max_seq_len, 1, 1, 15, 15]
                x = x.squeeze()
                x = x.contiguous().view(self.batch_size, self.max_seq_len, -1)  # [batch_size, max_seq_len, 225]
            ######################

        if self.ablation == "WEIGHT_SUM":
            # TODO ablation: fusion 模块，改为直接相加除2
            x = 0.5 * x + 0.5 * lstm_output
        else:
            # TODO fusion gate 模块，两个gate
            gate_input = torch.cat([x, lstm_output], dim=2)
            c3d_gate = self.sigmoid(self.fc_c3d_gate(gate_input)) # 0.7-0.9
            lstm_gate = self.sigmoid(self.fc_lstm_gate(gate_input)) # 0.5-0.7
            x = c3d_gate * x + lstm_gate * lstm_output

        if self.ablation == "FC_REAR":
            # TODO ablation: 取消掉LSTM_REAR
            pass
        else:
            # TODO LSTM_REAR
            x, _ = self.LSTM_REAR(x)

        x = x.contiguous().view(self.batch_size * self.max_seq_len, -1) # [b*max_seq_len, 225]
        output = self.fc_final(x) # [b*max_seq_len, 110]
        return output, None, None
