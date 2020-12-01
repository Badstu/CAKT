import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as n

class CCAFusion(nn.Module):
    def __init__(self):
        super(CCAFusion, self).__init__()
        
    
    def forward(self, c3d_feature, lstm_feature):
        
        pass