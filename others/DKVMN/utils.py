import json

import torch.nn.init
from torch.autograd import Variable


def variable(tensor, gpu):
    if gpu >= 0:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def save_checkpoint(state, track_list, filename):
    with open(filename + '.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename + '.model')


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
