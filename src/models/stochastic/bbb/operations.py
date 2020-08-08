import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

from src.models.stochastic.bbb.utils import kl_divergence

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
  