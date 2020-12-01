import torch
import torch.nn as nn
from utils import *


class MODEL(nn.Module):

    def __init__(self, n_question, x_embed_dim, hidden_dim, hidden_layers, dropout_rate=0.6, gpu=0):
        super(MODEL, self).__init__()
        self.n_question = n_question
        self.x_embed_dim = x_embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.gpu = gpu
        self.hidden_layers = hidden_layers

        self.x_embed = nn.Embedding(2 * self.n_question + 1, self.x_embed_dim, padding_idx=0)

        self.rnn = nn.LSTM(input_size=self.x_embed_dim, hidden_size=self.hidden_dim,
                           num_layers=hidden_layers, batch_first=True)

        self.predict_linear = nn.Linear(self.hidden_dim, self.n_question, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.constant_(self.predict_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.x_embed.weight)

    def forward(self, x_data, q_t_data, target):
        # Target size [B*L,1]
        # q_t size [B*L,1]

        batch_size = x_data.shape[0]
        seqlen = x_data.shape[1]

        init_h = variable(torch.randn(self.hidden_layers, batch_size, self.hidden_dim), self.gpu)
        init_c = variable(torch.randn(self.hidden_layers, batch_size, self.hidden_dim), self.gpu)

        ## (q,a) embedding
        x_embed_data = self.x_embed(x_data)

        ## lstm process
        lstm_out, final_status = self.rnn(x_embed_data, (init_h, init_c))
        # print("lstm_out=", lstm_out)
        ## lstm out size [B,L,E]
        ## Target size [B,L]
        ## prediction size [B*L,Q]
        lstm_out = lstm_out.contiguous()
        prediction = self.predict_linear(self.dropout(lstm_out.view(batch_size * seqlen, -1)))

        ## predict_slice[i] size [1*Q]
        prediction_slice = torch.chunk(prediction, batch_size * seqlen, 0)

        # prediction_1d size [B*L,1]
        q_t_data = q_t_data - 1
        fill_mask = q_t_data.lt(0)
        q_t_data = q_t_data.masked_fill(fill_mask, 0)

        prediction_1d = torch.cat([prediction_slice[i][:, q_t_data[i]] for i in range(batch_size * seqlen)], 0)

        target_1d = target  # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)  # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(prediction_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)
        print(loss)

        return loss, torch.sigmoid(filtered_pred), filtered_target
