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


class CAKT(nn.Module):
    def __init__(self, k_frames, input_dim, H, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device,
                 init_params=True):
        super(CAKT, self).__init__()
        self.k_frames = k_frames
        self.know_length = output_dim
        self.device = device
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

        # self.LSTM_RECENT = RNN(hidden_dim, hidden_dim, 1, batch_size, device)

        self.LSTM_REAR = RNN(hidden_dim, hidden_dim, num_layers, batch_size, device)

        self.global_pooling = nn.AdaptiveAvgPool3d((1, self.H, self.H))
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

        '''
        # TODO 对pc和rg进行onehot化
        past_counts = torch.floor(torch.log2(past_counts.float())).long()
        past_counts[past_counts < 0] = 0
        repeat_gaps = torch.floor(torch.log2(repeat_gaps.float())).long()
        eye_matrix = torch.eye(10).to(self.device) # pc_num == rg_num == 10
        
        past_counts_expand = past_counts.view(self.batch_size * self.max_seq_len)
        pc_one_hot = eye_matrix[past_counts_expand]
        pc_embed_output = pc_one_hot.view(self.batch_size, self.max_seq_len, -1)
        
        repeat_gaps_expand = repeat_gaps.view(self.batch_size * self.max_seq_len)
        rg_one_hot = eye_matrix[repeat_gaps_expand]
        rg_embed_output = rg_one_hot.view(self.batch_size, self.max_seq_len, -1)
        '''
        
        '''
        # TODO 对pc和rg进行embedding
        pc_embed_output = self.pc_embedding(past_counts) # [batch_size, max_seq_len, 50]
        rg_embed_output = self.rg_embedding(repeat_gaps) # [batch_size, max_seq_len, 50]
        '''

        # TODO 对x进行embedding
        embed_output = self.embedding(x)  # [batch_size, max_seq_len, 205] embed_dim=205

        # TODO 把embed_output和两个 onehot feature 拼接到一起，变成225维的向量
        # [batch_size, max_seq_len, 225]
        # input_features = torch.cat([embed_output, pc_embed_output, rg_embed_output], dim=2)
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

        # '''
        # TODO C3D模块
        cnn_3d_input = cnn_3d_input.contiguous().view(cnn_3d_input.shape[0] * cnn_3d_input.shape[1], self.k_frames, self.H, self.H)
        cnn_3d_input = cnn_3d_input.unsqueeze(1)  # [batch_size * clip_nums, 1, k_frames, 15, 15]
        input_feature = cnn_3d_input
        x = self.layer1(cnn_3d_input)
        mid_feature1 = x
        x = self.layer2(x)
        mid_feature2 = x
        x = self.layer3(x)
        mid_feature3 = x
        x = self.layer4(x)  # [batch_size * max_seq_len, 1, k_frames, 15, 15]
        mid_feature4 = x

        x = self.global_pooling(x) # [batch_size * max_seq_len, 1, 1, 15, 15]
        global_pool_feature = x
        x = x.squeeze()
        x = x.contiguous().view(self.batch_size, self.max_seq_len, -1)  # [batch_size, max_seq_len, 225]
        ######################
        # '''

        '''
        # TODO compare: 用LSTM_RECENT代替C3D模块（只处理最近16个frames）
        output = []
        for i in range(self.max_seq_len):
            lstm_recent_input = cnn_3d_input[:, i, :, :, :].contiguous().view(self.batch_size, self.k_frames, -1)
            _, (h, c) = self.LSTM_RECENT(lstm_recent_input)
            output.append(h)
        x = torch.cat([x for x in output], dim=0).transpose(0, 1) # [batch_size, max_seq_len, 225]
        ###############
        '''

        '''
        # TODO fusion 模块，一个gate，用C3D的结果和LSTM的结果concat起来，fc+sigmoid得到update_gate
        gate = torch.cat([x, lstm_output], dim=2)
        update_gate = self.sigmoid(self.fc_gate(gate))
        x = update_gate * x + (1 - update_gate) * lstm_output
        '''
        
        '''
        # TODO compare: fusion 模块，直接相加除2
        x = 0.5 * x + 0.5 * lstm_output
        '''

        # TODO fusion 模块，两个gate
        gate_input = torch.cat([x, lstm_output], dim=2)
        c3d_gate = self.sigmoid(self.fc_c3d_gate(gate_input)) # 0.7-0.9
        lstm_gate = self.sigmoid(self.fc_lstm_gate(gate_input)) # 0.5-0.7
        x = c3d_gate * x + lstm_gate * lstm_output

        # TODO LSTM_REAR
        x, _ = self.LSTM_REAR(x)
        x = x.contiguous().view(self.batch_size * self.max_seq_len, -1) # [b*max_seq_len, 225]
        output = self.fc_final(x) # [b*max_seq_len, 110]
        return output, None, (input_feature, mid_feature1, mid_feature2, mid_feature3, mid_feature4, global_pool_feature)

# if __name__ == '__main__':
#     batch_size = 2
#     k_frames = 4
#     max_seq_len = 50
#     input_dim = 220
#     H = 15
#     embed_dim = 225
#     hidden_dim = 225
#     num_layers = 1
#     output_dim = 110
#     device = torch.device('cpu')
#     model = CAKT(k_frames, input_dim, H, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device)

#     test_input = torch.randint(1, 110, (batch_size, max_seq_len))
#     # output, _ = model(test_input)
#     # print(output.shape)

#     ef = ExtractForget(110, 2)
#     output_1 = ef.extract_repeated_time_gap(test_input[0], len(test_input[0]))
#     output_2 = ef.extract_past_trail_counts(test_input[0], len(test_input[0]))
#     print(output_1, output_2)
