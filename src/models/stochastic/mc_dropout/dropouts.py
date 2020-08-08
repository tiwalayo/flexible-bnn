import torch.nn.functional as F
import torch 
from torch.autograd import Variable
import torch.nn as nn

DROPOUT_FACTORY = {'gaussian': lambda p: GaussianDropout(p),
                   'bernoulli': lambda p: BernoulliDropout(p)}

class GaussianDropout(nn.Module):
    def __init__(self, p=0.0):
        super(GaussianDropout, self).__init__()
        self.p = torch.Tensor([p/(1-p)])
        
    def forward(self, x):
        epsilon = Variable(torch.randn(x.size()) * self.p + 1.0)
        if x.is_cuda:
            epsilon = epsilon.cuda()
        return x * epsilon

class BernoulliDropout(nn.Module):
    def __init__(self, p=0.0):
        super(BernoulliDropout, self).__init__()
        self.p = p
        
    def forward(self, x):
        if len(x.shape)>2:
            return F.dropout2d(x, p=self.p, training=True, inplace=False)
        else:
            return F.dropout(x, p=self.p, training=True, inplace=False)