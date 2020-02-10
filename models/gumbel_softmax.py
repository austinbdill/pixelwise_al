import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps)).cuda()

def gumbel_softmax_sample(logits, temperature, noisy):
    y = logits
    if noisy:
        y = y + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, noisy=True):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature ,noisy)
    shape = y.size()
    _, ind = y.max(dim=-1)
    #y_soft = y[:, -1]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    proxy = (y_hard - y).detach() + y
    return proxy[:, -1]
    #return y_soft, y_hard
