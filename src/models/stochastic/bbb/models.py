import torch.nn as nn
import torch.nn.functional as F
import torch 

from src.models.stochastic.bbb.operations import Linear, Conv2d


class LinearNetwork(nn.Module):
  def __init__(self, input_size, output_size, layers, activation, args):
    super(LinearNetwork, self).__init__()
    self.args = args
    self.input_size = 1

    for i in input_size:
        self.input_size*=int(i) 
    self.output_size = int(output_size)

    if activation == "relu":
        self.activation = nn.ReLU
    elif activation == "tanh":
        self.activation = nn.Tanh
    elif activation == "linear":
        self.activation = nn.Identity
    else:
        raise NotImplementedError("Other activations have not been implemented!")

    self.layers = nn.ModuleList()
    for i in range(len(layers)):
        if i == 0:
            self.layers.append(Linear(self.input_size, int(layers[0]),sigma_prior=args.sigma_prior,  bias=False))
        else:
            self.layers.append(Linear(int(layers[i-1]), int(layers[i]), sigma_prior=args.sigma_prior,  bias=False))
        self.layers.append(self.activation())

    self.layers.append(Linear(int(layers[len(layers)-1]), self.output_size, sigma_prior=args.sigma_prior,  bias=False))
    
    
  def forward(self, input):
    x = input.view(-1,self.input_size)
    kl = 0.0
    for i, layer in enumerate(self.layers):
        if isinstance(layer, Linear):
            x, _kl = layer(x)
            kl+=_kl
        else:
            x = layer(x)
    if self.args.task == "binary_classification":
        x = torch.sigmoid(x)
    elif self.args.task == "classification":
        x = F.softmax(x, dim=-1)
    return x, kl.view(1)
  

  def log(self, *args, **kwargs):
    pass
  

class ConvNetwork(nn.Module):
    def __init__(self, input_size, output_size, layers, activation, args):
        super(ConvNetwork, self).__init__()
        self.args = args 

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "linear":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError("Other activations have not been implemented!")
        self.conv1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(5,5), stride=1, padding=2, sigma_prior=args.sigma_prior,  bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5), stride=1, padding=2, sigma_prior=args.sigma_prior,  bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = Linear(in_features=50*7*7, out_features=500, sigma_prior=args.sigma_prior,  bias=False)
        self.linear2 = Linear(in_features=500, out_features=output_size, sigma_prior=args.sigma_prior,  bias=False)


    def forward(self, x):
        kl = 0.0
        x, _kl = self.conv1(x)
        kl+=_kl
        x = self.pool1(x)
        x, _kl = self.conv2(x)
        kl+=_kl
        x = self.pool2(x)
        x = x.view(x.size(0),-1)
        x, _kl = self.linear1(x)
        kl+=_kl
        x = self.activation(x)
        x, _kl = self.linear2(x)
        kl+=_kl
        x = F.softmax(x, dim=-1)
        return x, kl.view(1)

    def log(self, *args, **kwargs):
        pass
  
