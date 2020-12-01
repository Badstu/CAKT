'''
2020/10/04
CKT_dev is CKT model with two memory block
'''

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


class MemoryBlock(nn.Module):
    '''
    @ input: Attention!!! the input of memory block is one batch one item in sequence
    '''
    def __init__(self, knowledge_length, concept_length, knowledge_emb_size, interaction_emb_size, batch_size, device):
        '''
        @params:
        knowledge_length: 110
        concept_length: 20
        knowledge_emb_size: 100
        interaction_emb_size: 200
        '''
        super(MemoryBlock, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.knowledge_length = knowledge_length
        self.concept_length = concept_length
        self.knowledge_emb_size = knowledge_emb_size
        self.interaction_emb_size = interaction_emb_size

        # key memory [concept_length, know_emb_size]
        self.question_embedding = nn.Embedding(knowledge_length + 1, self.knowledge_emb_size, padding_idx=0)
        self.key_memory = nn.Parameter(torch.zeros(self.concept_length, self.knowledge_emb_size))
        self.key_memory.data.uniform_(-1, 1)

        # value memory [batch_size, know_len, inter_emb_size]
        self.interaction_embedding = nn.Embedding(2 * knowledge_length + 1, self.interaction_emb_size, padding_idx=0)

        self.tanh = nn.Tanh()


    def forward(self, x, value_memory):
        '''
        x: [batch, 1] 一整个batch的当前时刻的item
        value_memory = torch.zeros(self.batch_size, self.knowledge_length, self.interaction_emb_size))
        value_memory.data.uniform_(-1, 1)
        '''
        label = torch.ones_like(x)
        label[x <= self.knowledge_length] = 0
        question_id = (x-1) % self.knowledge_length + 1

        # two types embedding
        question_token = self.question_embedding(question_id)
        interaction_token = self.interaction_embedding(x) # [batch_size, 1, int_emb_size]
        interaction_token = self.tanh(interaction_token)

        # calculate softmax weight
        question_token = question_token.squeeze().transpose(1, 0)
        weight = torch.mm(self.key_memory, question_token).T #[batch_size, concept_length]
        weight = nn.functional.softmax(weight, dim=-1)

        # expand weight and interaction token
        ex_weight = weight.unsqueeze(2).expand(-1, -1, self.interaction_emb_size) #[batch_size, concept_length, inter_emb_size]
        ex_interaction = interaction_token.expand(-1, self.concept_length, -1) #[batch_size, concept_length, inter_emb_size]

        # update value memory
        update_information = torch.mul(ex_interaction, ex_weight)
        value_memory = value_memory + update_information
        return value_memory # [batch_size, concept_length, interaction_emb_size]


class CKT_dev(nn.Module):
    # def __init__(self, k_frames, input_dim, H, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device,
    #              init_params=True):
    def __init__(self, k_frames, knowledge_length, concept_length, knowledge_emb_size, interaction_emb_size, lstm_hidden_dim, lstm_num_layers, batch_size, device, init_params=True):
        super(CKT_dev, self).__init__()
        self.k_frames = k_frames
        self.knowledge_length = knowledge_length
        self.concept_length = concept_length
        self.knowledge_emb_size = knowledge_emb_size
        self.interaction_emb_size = interaction_emb_size

        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.batch_size = batch_size
        self.device = device
        self.inplanes = 1
        # self.H = H
        
        self.lstm_interaction_embedding = nn.Embedding(2 * self.knowledge_length + 1, self.interaction_emb_size, padding_idx=0)

        self.memory_model = MemoryBlock(self.knowledge_length,
                                        self.concept_length,
                                        self.knowledge_emb_size,
                                        self.interaction_emb_size,
                                        self.batch_size,
                                        self.device)

        self.layer1 = self._make_layer(C3DBlock, 4, 1)
        # self.layer2 = self._make_layer(C3DBlock, 8, 1)
        self.layer3 = self._make_layer(C3DBlock, 4, 1)
        self.layer4 = self._make_layer(C3DBlock, 1, 1)
        
        self.LSTM_FRONT = RNN(self.interaction_emb_size, self.lstm_hidden_dim, self.lstm_num_layers, self.batch_size, self.device)
        self.LSTM_REAR = RNN(self.lstm_hidden_dim, self.lstm_hidden_dim, self.lstm_num_layers, self.batch_size, self.device)
        self.fc_c3d_gate = nn.Linear(2 * self.lstm_hidden_dim, self.lstm_hidden_dim, bias=True)
        self.fc_lstm_gate = nn.Linear(2 * self.lstm_hidden_dim, self.lstm_hidden_dim, bias=True)
        # self.fc_final = nn.Linear(self.lstm_hidden_dim, 1, bias=True)
 
        self.global_pooling = nn.AdaptiveAvgPool3d((1, self.concept_length, self.interaction_emb_size))
        self.final_layer = nn.Linear(self.interaction_emb_size, 1, bias=True)

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

    def forward(self, x, next_question):
        '''
        x: [batch_size, max_seq_len] ([2, 143] item > 110)
        next_question: [batch_size, max_seq_len, (next_qnumber, next_0_1)]
        '''
        # print(x.shape)
        self.batch_size = x.shape[0]
        self.max_seq_len = x.shape[1]
        self.clips_nums = self.max_seq_len - self.k_frames + 1
        # self.clips_nums = self.max_seq_len

        # TODO lstm 模块
        embed_output = self.lstm_interaction_embedding(x) # [batch_size, max_seq_len, 50], interaction_emb_size=50
        lstm_output, _ = self.LSTM_FRONT(embed_output) # [batch_size, max_seq_len, 50]

        # TODO 初始化value_memory，每一个学生（记录）需要一个value_memory
        value_memory = torch.zeros((self.batch_size, self.concept_length, self.interaction_emb_size)).to(self.device)
        value_memory.data.uniform_(-1, 1)

        tmp = torch.zeros((self.batch_size, 1, self.concept_length, self.interaction_emb_size)).to(self.device)

        cnn_3d_output = []
        # TODO 利用memory block把输入变成一个矩阵形式 [batch_size, 8, concept_length, knowledge_embedding_size]
        for item_index in range(self.max_seq_len):
            batch_item = x[:, item_index:item_index+1]
            value_memory = self.memory_model(batch_item, value_memory=value_memory)

            # 0-6, 第7个frame可以用0-7共8个frame拼成一个clip
            if item_index < self.k_frames - 1:
                tmp = torch.cat((tmp, value_memory.unsqueeze(1)), dim=1)
                zero_padding = torch.zeros((self.batch_size, self.k_frames - item_index - 1, self.concept_length,
                                            self.interaction_emb_size)).to(self.device)
                # [batch_size, 8, 110, 100]
                ttmp = torch.cat((zero_padding, tmp[:, 1:]), dim=1)
                cnn_3d_input = ttmp
            else:
                tmp = torch.cat((tmp, value_memory.unsqueeze(1)), dim=1)
                tmp = tmp[:, -1 * self.k_frames:]
                cnn_3d_input = tmp

            # cnn_3d_input = torch.cat([x for x in cnn_3d_input], 1) #[batch_size, max_seq_len, k, k_len, i_emb_size]
            ##############

            # TODO C3D模块
            cnn_3d_input = cnn_3d_input.contiguous().view(self.batch_size, self.k_frames,
                                                          self.concept_length, self.interaction_emb_size)
            cnn_3d_input = cnn_3d_input.unsqueeze(1)  # [batch_size, 1, k_frames, 110, 100]
            tmp_x = self.layer1(cnn_3d_input)
            # tmp_x = self.layer2(tmp_x)
            tmp_x = self.layer3(tmp_x)
            tmp_x = self.layer4(tmp_x)  # [batch_size, 1, k_frames, 110, 100]

            tmp_x = self.global_pooling(tmp_x)  # [batch_size, 1, 1, 110, 100]
            tmp_x = tmp_x.squeeze() # [batch_size, 110, 100]
            ######################

            # TODO calculate next question weight
            next_question_number = next_question[:, item_index:item_index+1, 0].view(-1).long()
            next_question_label = next_question[:, item_index:item_index+1, 1].view(-1)

            next_question_token = self.memory_model.question_embedding(next_question_number)
            next_question_token = next_question_token.transpose(1, 0)

            # weight: [batch_size, 110]
            next_question_weight = torch.mm(self.memory_model.key_memory, next_question_token).T
            next_question_weight = nn.functional.softmax(next_question_weight, dim=-1)
            next_question_weight = next_question_weight.unsqueeze(2)

            # TODO calculate next prediction score
            s = (next_question_weight * tmp_x).sum(dim=1) # [batch_size, interaction_emb_size]
            # predict_score = self.final_layer(s)
            cnn_3d_output.append(s.unsqueeze(1))

        cnn_3d_output = torch.cat([i for i in cnn_3d_output], dim=1) # [batch_size, max_seq_len, 50] inter_emb_size = 50

        # TODO 用C3D的结果和LSTM的结果concat起来，fc+sigmoid得到update_gate, update_gate是一个vector，长度是50
        gate_input = torch.cat([cnn_3d_output, lstm_output], dim=2)
        c3d_gate = self.sigmoid(self.fc_c3d_gate(gate_input))
        lstm_gate = self.sigmoid(self.fc_lstm_gate(gate_input)) 
        x = c3d_gate * cnn_3d_output + lstm_gate * lstm_output

        # TODO LSTM_REAR
        x, _ = self.LSTM_REAR(x)
        x = x.contiguous().view(self.batch_size, self.max_seq_len, -1) # [batch_size, max_seq_len, 50]
        output = self.final_layer(x) # [batch_size, max_seq_len, 1]
        output = output.squeeze(dim=2)
        return output

if __name__ == '__main__':
    knowledge_length = 110
    concept_length = 20
    knowledge_emb_size = 20
    interaction_emb_size = 50
    lstm_hidden_dim = 50
    lstm_num_layers = 1

    batch_size = 8
    device = torch.device("cpu")
    k_frames = 4
    max_seq_len = 23

    # model = MemoryBlock(batch_size, knowledge_length, knowledge_emb_size, interaction_emb_size, device)
    model = CKT_dev(k_frames, knowledge_length, concept_length, knowledge_emb_size, interaction_emb_size, lstm_hidden_dim,
                 lstm_num_layers, batch_size, device)
    model = model.to(device)

    x = torch.randint(1, 220, (batch_size, 13))
    print("input shape is: ", x.shape)
    query_question = torch.randint(1, 110, (batch_size, 13, 2))
    query_question[:, :, 1] = 0
    model(x, query_question)

