import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np


class RNN_DKT(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, output_dim, batch_size, device):
        super(RNN_DKT, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        self.embedding = nn.Embedding(input_dim + 1, embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.6)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)


    def forward(self, x):
        h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        x_embed = self.embedding(x)
        # batch_seq_pack = rnn_utils.pack_padded_sequence(x_embed, batch_len, batch_first=True)
        
        output, hidden_state = self.LSTM(x_embed, (h0, c0))
        output = output.contiguous().view(output.shape[0] * output.shape[1], -1)
        output = self.fc(self.dropout(output))
        return output, hidden_state

    def init_params(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.kaiming_normal_(self.embedding.weight)

