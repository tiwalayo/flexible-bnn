import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.tensor as Tensor
import numpy as np
import matplotlib.pyplot as plt

from src.models.stochastic.bbb.utils import kl_divergence, normpdf, KumaraswamyKL

class Conv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1,
               bias=False, sigma_prior=-2):

    super(Conv2d, self).__init__(in_channels, out_channels,kernel_size, stride, padding, dilation, groups, bias)

    self.log_sigma = torch.nn.Parameter(
            torch.ones_like(self.weight)*(-10.), requires_grad=True)
    self.log_sigma_prior = sigma_prior

  def forward(self, X):
    Z_mean = F.conv2d(X, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
    Z_std = torch.sqrt(1e-8+F.conv2d(torch.pow(X, 2), torch.pow(F.softplus(self.log_sigma), 2),
                         None, self.stride, self.padding, self.dilation, self.groups))
    Z_noise = torch.ones_like(Z_mean).normal_(0, 1).to(X.device)
    Z = Z_mean + Z_std * Z_noise 

    kl = kl_divergence(self.weight, F.softplus(self.log_sigma),
                       torch.zeros_like(self.weight).to(self.weight.device), 
                       F.softplus(torch.ones_like(self.log_sigma)*self.log_sigma_prior).to(self.weight.device))
    return Z, kl 

class Linear(nn.Linear):
  
  def __init__(self, in_features, out_features, bias=False, sigma_prior=-2):
    super().__init__(in_features, out_features, bias)
    self.log_sigma = torch.nn.Parameter(torch.ones_like(self.weight)*(-5), requires_grad=True)
    self.log_sigma_prior = sigma_prior
  

  def forward(self, X):
    Z_mean = torch.mm(X, self.weight.t())
    Z_std = torch.sqrt(1e-8+torch.mm(torch.pow(X,2), torch.pow(F.softplus(self.log_sigma).t(), 2))) 
    Z_noise = torch.ones_like(Z_mean).normal_(0, 1).to(X.device)
    
    Z = Z_mean + Z_std * Z_noise

    kl = kl_divergence(self.weight, F.softplus(self.log_sigma),
                       torch.zeros_like(self.weight).to(self.weight.device), 
                       F.softplus(torch.ones_like(self.log_sigma)*self.log_sigma_prior).to(self.weight.device))
    return Z, kl

class Kumaraswamy(nn.Linear):
  
  def __init__(self, in_features, out_features, bias=False):
    super().__init__(in_features, out_features, bias)
    # `weight` is log a, `log_b` is log b
    self.log_b = torch.nn.Parameter(torch.ones_like(self.weight)*(5), requires_grad=True)
  

  def forward(self, X):
    eps = torch.ones_like(self.weight).uniform_(0, 1).to(X.device)
    T_ = lambda x: 2*torch.pow(1 - torch.pow(1-x,1/F.softplus(self.log_b)), 1/F.softplus(self.weight))-1
    
    weightsample = T_(eps)
    Z = torch.mm(X, weightsample.t())

    GAMMA = 0.57721566490153286060651209008240243104215933593992
    kl = -((1-1/F.softplus(self.log_b)) + (1-1/F.softplus(self.weight)) * (GAMMA + torch.log(1e-20+F.softplus(self.log_b))) - torch.log(1e-20+F.softplus(self.weight) * F.softplus(self.log_b))).sum()
    
    return Z, kl

class FNetLinear(nn.Linear):
  
  def __init__(self, in_features, out_features, bias=False):
    super().__init__(in_features, out_features, bias)
    # `weight` is log a, `log_b` is log b
    self.nn = FNet(in_features, out_features)
  

  def forward(self, X):
    eps = torch.ones_like(self.weight).uniform_(0, 1).requires_grad_(True).to(X.device)
    self.nn.regularize()
    weightsample = self.nn(eps.view(-1)).view(eps.shape[0], eps.shape[1])
    
    Z = torch.mm(X, weightsample.t())

    F = self.nn.F(x)
    qpF = torch.autograd.grad(self.nn(F), F, create_graph=True, allow_unused = True, grad_outputs=torch.ones_like(samples))
    kl = -torch.log(qpF).sum()/(eps.shape[0]*eps.shape[1])
    
    self.nn.optimizer.zero_grad()
    kl.backward()
    self.nn.optimizer.step()
    
    return Z, kl

class FNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = nn.Linear(in_features*out_features, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, in_features*out_features)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        
    def forward(self, x_):
        x = self.sigmoid(self.linear1(x_))
        x = self.sigmoid(self.linear2(x))
        x = self.linear3(x)
        x[x_ < 1e-5] = -200.
        x[x_ > 1-1e-5] = 200.
        return x
    
    def getDLoss(self, n_samples=5):
        
        def set_grad(var):
            def hook(grad):
                var.grad = grad
            return hook

        samples = Tensor(np.random.uniform(1e-5,1-1e-5,(n_samples,self.out_features*self.in_features))).float().requires_grad_(True).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu' ))
        samples.retain_grad()
        sampled = self(samples)
        grads = torch.autograd.grad(sampled, samples, create_graph=True, allow_unused = True, grad_outputs=torch.ones_like(samples))
        dloss = self.relu(-grads[0])
        return dloss.sum()
    
    def F(self, x):
        
        low = torch.empty(x.shape).fill_(1e-6).to(x.device)
        high = torch.empty(x.shape).fill_(1-1e-6).to(x.device)
        
        func = lambda t: self(t)-x
        
        def samesign(a, b):
            return a*b>0
        
        for i in range(54):
            midpoint = (low + high) / 2.0
            if samesign(func(low), func(midpoint)):
                low = midpoint
            else:
                high = midpoint

        return midpoint
    
    def regularize(self):
        self.optimizer.zero_grad()
        loss = self.getDLoss(n_samples=50)
        loss.backward()
        self.optimizer.step()