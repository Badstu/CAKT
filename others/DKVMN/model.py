import torch
import torch.nn as nn
import numpy as np
import time
from memory import DKVMN
import numpy
from itertools import zip_longest

class MODEL(nn.Module):

    def __init__(self, n_question, batch_size, q_embed_dim, qa_embed_dim,
                 memory_size, memory_key_state_dim, memory_value_state_dim, final_fc_dim, gpu, student_num=None):
        super(MODEL, self).__init__()
        self.n_question = n_question
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.final_fc_dim = final_fc_dim
        self.student_num = student_num

        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.q_embed_dim, self.final_fc_dim, bias=True)
        # self.predict_linear = nn.Linear(self.memory_value_state_dim + self.q_embed_dim, 1, bias=True)
        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal_(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_value)

        # modify hop_lstm
        self.hop_lstm = nn.LSTM(input_size=self.memory_value_state_dim + self.q_embed_dim, hidden_size=64, num_layers=1, batch_first=True)
        # hidden_size = 64
        self.predict_linear = nn.Linear(64, 1, bias=True)

        self.mem = DKVMN(memory_size=self.memory_size,
                   memory_key_state_dim=self.memory_key_state_dim,
                   memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        # 题目序号从1开始
        # nn.embedding输入是一个下标的列标，输出是对应的嵌入
        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        # self.a_embed = nn.Linear(2 * self.n_question + 1, self.qa_embed_dim, bias=True)
        self.a_embed = nn.Linear(self.final_fc_dim + 1, self.qa_embed_dim, bias=True)

        self.correlation_weight_list = []

        if gpu >= 0:
            self.device = torch.device('cuda', gpu)
        else:
            self.device = torch.device('cpu')

        print("num_layers=1, final=110, a=0.075, b=0.088, c=1.00")

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed.weight)

    # 方法2：权重向量的topk置1
    def identity_layer(self, correlation_weight, k=1):
        batch_identity_indices = []

        # 把batch中每一格sequence中topk置1，其余置0
        _, indices = correlation_weight.topk(k, dim=1, largest=True)
        identity_matrix = torch.zeros([self.batch_size, self.memory_size])
        for i, m in enumerate(indices):
            identity_matrix[i, m] = 1

        identity_matrix = torch.chunk(identity_matrix, self.batch_size, 0)

        for identity_vector in identity_matrix:

            identity_vector = list(identity_vector.squeeze(0))

            # 获取每一个sequence对应的identity向量的标号
            if identity_vector not in self.correlation_weight_list:
                self.correlation_weight_list.append(identity_vector)
                # index = len(self.correlation_weight_list)-1
                # index = self.correlation_weight_list.index(identity_vector)
                batch_identity_indices.append(len(self.correlation_weight_list)-1)
            else:
                # index = self.correlation_weight_list.index(identity_vector)
                batch_identity_indices.append(self.correlation_weight_list.index(identity_vector))

        batch_identity_indices = torch.tensor(batch_identity_indices)
        return batch_identity_indices

    # 方法1：用三角隶属函数计算identity向量
    def triangular_layer(self, correlation_weight, seqlen, a=0.075, b=0.088, c=1.00):
        # batch_identity_indices = []

        # w'= max((w-a)/(b-a), (c-w)/(c-b))
        # min(w', 0)
        correlation_weight = correlation_weight.view(self.batch_size * seqlen, -1)
        correlation_weight = torch.cat([correlation_weight[i] for i in range(correlation_weight.shape[0])], 0).unsqueeze(0)
        correlation_weight = torch.cat([(correlation_weight-a)/(b-a), (c-correlation_weight)/(c-b)], 0)
        correlation_weight, _ = torch.min(correlation_weight, 0)
        w0 = torch.zeros(correlation_weight.shape[0]).to(self.device)
        correlation_weight = torch.cat([correlation_weight.unsqueeze(0), w0.unsqueeze(0)], 0)
        correlation_weight, _ = torch.max(correlation_weight, 0)

        identity_vector_batch = torch.zeros(correlation_weight.shape[0]).to(self.device)

        # >=0.6的值置2，0.1-0.6的值置1，0.1以下的值置0
        # mask = correlation_weight.lt(0.1)
        identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.lt(0.1), 0)
        # mask = correlation_weight.ge(0.1)
        identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.ge(0.1), 1)
        # mask = correlation_weight.ge(0.6)
        _identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.ge(0.6), 2)

        # identity_vector_batch = torch.chunk(identity_vector_batch.view(self.batch_size, -1), self.batch_size, 0)

        # TODO 矩阵化上述代码
        # 输入：_identity_vector_batch
        # 输出：indices
        identity_vector_batch = _identity_vector_batch.view(self.batch_size * seqlen, -1)

        unique_iv = torch.unique(identity_vector_batch, sorted=False, dim=0)
        self.unique_len = unique_iv.shape[0]

        # A^2
        iv_square_norm = torch.sum(torch.pow(identity_vector_batch, 2), dim=1, keepdim=True)
        iv_square_norm = iv_square_norm.repeat((1, self.unique_len))
        # B^2.T
        unique_iv_square_norm = torch.sum(torch.pow(unique_iv, 2), dim=1, keepdim=True)
        unique_iv_square_norm = unique_iv_square_norm.repeat((1, self.batch_size * seqlen)).transpose(1, 0)
        # A * B.T
        iv_matrix_product = identity_vector_batch.mm(unique_iv.transpose(1, 0))
        # A^2 + B^2 - 2A*B.T
        iv_distances = iv_square_norm + unique_iv_square_norm - 2 * iv_matrix_product
        indices = (iv_distances == 0).nonzero()

        batch_identity_indices = indices[:, -1]
        print(self.unique_len)
        print(batch_identity_indices.shape)
        return batch_identity_indices


    '''
    # 方法1：用三角隶属函数计算identity向量
    def triangular_layer(self, correlation_weight, a=0.075, b=0.088, c=1.00):
        batch_identity_indices = []

        # w'= max((w-a)/(b-a), (c-w)/(c-b))
        # min(w', 0)
        correlation_weight = torch.cat([correlation_weight[i] for i in range(self.batch_size)], 0).unsqueeze(0)
        correlation_weight = torch.cat([(correlation_weight-a)/(b-a), (c-correlation_weight)/(c-b)], 0)
        correlation_weight, _ = torch.min(correlation_weight, 0)
        w0 = torch.zeros(correlation_weight.shape[0]).to(self.device)
        correlation_weight = torch.cat([correlation_weight.unsqueeze(0), w0.unsqueeze(0)], 0)
        correlation_weight, _ = torch.max(correlation_weight, 0)

        identity_vector_batch = torch.zeros(correlation_weight.shape[0]).to(self.device)

        # >=0.6的值置2，0.1-0.6的值置1，0.1以下的值置0
        # mask = correlation_weight.lt(0.1)
        identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.lt(0.1), 0)
        # mask = correlation_weight.ge(0.1)
        identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.ge(0.1), 1)
        # mask = correlation_weight.ge(0.6)
        _identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.ge(0.6), 2)

        # TODO 矩阵化上述代码
        # 输入：_identity_vector_batch
        # 输出：indices
        identity_vector_batch = _identity_vector_batch.view(self.batch_size, -1)

        unique_iv = torch.unique(identity_vector_batch, dim = 0)
        self.unique_len = unique_iv.shape[0]

        # A^2
        iv_square_norm = torch.sum(torch.pow(identity_vector_batch, 2), dim=1, keepdim=True)
        iv_square_norm = iv_square_norm.repeat((1, self.unique_len))
        # B^2.T
        unique_iv_square_norm = torch.sum(torch.pow(unique_iv, 2), dim=1, keepdim=True)
        unique_iv_square_norm = unique_iv_square_norm.repeat((1, self.batch_size)).transpose(1, 0)
        # A * B.T
        iv_matrix_product = identity_vector_batch.mm(unique_iv.transpose(1, 0))
        # A^2 + B^2 - 2A*B.T
        iv_distances = iv_square_norm + unique_iv_square_norm - 2 * iv_matrix_product
        indices = (iv_distances == 0).nonzero()
        batch_identity_indices = indices[:, -1]

        return batch_identity_indices
        
        #batch_size = 32, seq_len = 200 memory_size = 20, memory_key_state_dim = 50, memory_value_state_dim=200
        print(identity_vector_batch.shape)
        identity_vector_batch = torch.chunk(_identity_vector_batch.view(self.batch_size, -1), self.batch_size, 0)

        print(len(identity_vector_batch))

        for identity_vector in identity_vector_batch:

            identity_vector = list(identity_vector.squeeze(0))

            # 获取每一个sequence对应的identity向量的标号
            if identity_vector not in self.correlation_weight_list:
                self.correlation_weight_list.append(identity_vector)
                # index = len(self.correlation_weight_list)-1
                # index = self.correlation_weight_list.index(identity_vector)
                batch_identity_indices.append(len(self.correlation_weight_list) - 1)
            else:
                # index = self.correlation_weight_list.index(identity_vector)
                batch_identity_indices.append(self.correlation_weight_list.index(identity_vector))

        batch_identity_indices = torch.tensor(batch_identity_indices)
    '''


    def forward(self, q_data, qa_data, a_data, target, student_id=None):

        batch_size = q_data.shape[0]   #32
        seqlen = q_data.shape[1]   #200

        self.correlation_weight_list.clear()

        ## qt && (q,a) embedding
        q_embed_data = self.q_embed(q_data)

        # modify 生成每道题对应的yt onehot向量
        # a_onehot_array = []
        # for i in range(a_data.shape[0]):
        #     for j in range(a_data.shape[1]):
        #         a_onehot = np.zeros(111)
        #         index = a_data[i][j]
        #         if index > 0:
        #             a_onehot[index] = 1
        #         a_onehot_array.append(a_onehot)
        # a_onehot_content = torch.cat([torch.Tensor(a_onehot_array[i]).unsqueeze(0) for i in range(len(a_onehot_array))], 0)
        # a_onehot_content = a_onehot_content.view(batch_size, seqlen, -1).to(self.device)


        ## copy mk batch times for dkvmn
        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        ## slice data for seqlen times by axis 1
        slice_q_data = torch.chunk(q_data, seqlen, 1)
        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)

        # modify
        # slice_a_onehot_content = torch.chunk(a_onehot_content, seqlen, 1)
        slice_a = torch.chunk(a_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        identity_index_list = []
        correlation_weight_list = []

        # modify
        f_t = []

        # (n_layers，batch_size，hidden_dim)
        init_h = torch.randn(1, self.batch_size, 64).to(self.device)
        init_c = torch.randn(1, self.batch_size, 64).to(self.device)

        time_start = time.time()

        for i in range(seqlen):
            ## Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)

            ## Read Process
            read_content = self.mem.read(correlation_weight)

            '''
            # modify get identity vector index （方法1和2）
            # 三角隶属函数方法
            identity_index = self.triangular_layer(correlation_weight)
            # topk置1
            # identity_index = self.identity_layer(correlation_weight)

            identity_index_list.append(identity_index)
            '''

            # modify
            correlation_weight_list.append(correlation_weight)

            ## save intermedium data
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            # modify
            batch_predict_input = torch.cat([read_content, q], 1)
            f = self.read_embed_linear(batch_predict_input)
            f_t.append(batch_predict_input)

            # 写入value矩阵的输入为[yt, ft]，onehot向量和ft向量拼接
            # onehot = slice_a_onehot_content[i].squeeze(1)
            # write_embed = torch.cat([onehot, f], 1)

            # 写入value矩阵的输入为[ft, yt]，ft直接和题目对错（0或1）拼接
            write_embed = torch.cat([f, slice_a[i].float()], 1)

            write_embed = self.a_embed(write_embed)
            new_memory_value = self.mem.write(correlation_weight, write_embed)

        # time_end = time.time()
        print("memory part:" + str(time.time() - time_start))

        # modify
        correlation_weight_matrix = torch.cat([correlation_weight_list[i].unsqueeze(1) for i in range(seqlen)], 1)
        identity_index_list = self.triangular_layer(correlation_weight_matrix, seqlen)
        # identity_index_list = self.identity_layer(correlation_weight_matrix, seqlen)
        identity_index_list = identity_index_list.view(self.batch_size, seqlen)

        print("tr part:" + str(time.time() - time_start))

        f_t = torch.cat([f_t[i].unsqueeze(1) for i in range(seqlen)], 1)
        target_seqlayer = target.view(batch_size, seqlen, -1)

        target_sequence = []
        pred_sequence = []

        time_start = time.time()

        for idx in range(self.unique_len):
            hop_lstm_input = []
            hop_lstm_target = []
            max_seq = 0
            for i in range(self.batch_size):
                # 获取每个sequence中和当前要进行预测的identity向量对应的题目在矩阵中的index
                index = list((identity_index_list[i,:]==idx).nonzero())
                max_seq = max(max_seq, len(index))
                if len(index) == 0:
                    hop_lstm_input.append(torch.zeros([1, self.memory_value_state_dim + self.q_embed_dim]))
                    hop_lstm_target.append(torch.full([1, 1], -1))
                    continue
                else:
                    index = torch.cat([torch.LongTensor(index[i]) for i in range(len(index))], 0).to(self.device)
                    hop_lstm_target_slice = torch.index_select(target_seqlayer[i, :, :], 0, index)
                    hop_lstm_input_slice = torch.index_select(f_t[i, :, :], 0, index)
                    # hop_lstm_input_slice = hop_lstm_input_slice.unsqueeze(0)
                    hop_lstm_input.append(hop_lstm_input_slice)
                    hop_lstm_target.append(hop_lstm_target_slice)

            # 给输入矩阵和target矩阵做padding
            for i in range(self.batch_size):
                x = torch.zeros([max_seq, self.memory_value_state_dim + self.q_embed_dim])
                x[:len(hop_lstm_input[i]), :] = hop_lstm_input[i]
                hop_lstm_input[i] = x
                y = torch.full([max_seq, 1], -1)
                y[:len(hop_lstm_target[i]), :] = hop_lstm_target[i]
                hop_lstm_target[i] = y


            # hop lstm进行预测
            hop_lstm_input = torch.cat([hop_lstm_input[i].unsqueeze(0) for i in range(self.batch_size)], 0).to(self.device)
            hop_lstm_target = torch.cat([hop_lstm_target[i].unsqueeze(0) for i in range(self.batch_size)], 0)

            hop_lstm_output, _ = self.hop_lstm(hop_lstm_input, (init_h, init_c))
            pred = self.predict_linear(hop_lstm_output)
            pred = pred.view(self.batch_size * max_seq, -1)
            hop_lstm_target = hop_lstm_target.view(self.batch_size * max_seq, -1).to(self.device)
            mask = hop_lstm_target.ge(0)
            hop_lstm_target = torch.masked_select(hop_lstm_target, mask)
            pred = torch.sigmoid(torch.masked_select(pred, mask))
            target_sequence.append(hop_lstm_target)
            pred_sequence.append(pred)

            # 在训练阶段对每个identity向量对应的lstm分别进行反向传播
            if self.training is True:
                subsequence_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, hop_lstm_target)
                subsequence_loss.backward(retain_graph=True)

        time_end = time.time()
        print("predict part:" + str(time_end - time_start))

        # 计算一个batch全部题目的loss
        target_sequence = torch.cat([target_sequence[i] for i in range(len(target_sequence))], 0)
        pred_sequence = torch.cat([pred_sequence[i] for i in range(len(pred_sequence))], 0)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_sequence, target_sequence)

        return loss, pred_sequence, target_sequence