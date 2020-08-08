import torch.nn as nn
import torch.nn.functional as F
import torch 

from src.models.stochastic.mc_dropout.dropouts import DROPOUT_FACTORY


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
            self.layers.append(nn.Linear(self.input_size, int(layers[0]),  bias=False))
        else:
            self.layers.append(nn.Linear(int(layers[i-1]), int(layers[i]),  bias=False))
        self.layers.append(self.activation())
        self.layers.append(DROPOUT_FACTORY[self.args.dropout_type](self.args.p))


    self.layers.append(nn.Linear(int(layers[len(layers)-1]), self.output_size,  bias=False))
    
    
  def forward(self, input):
    x = input.view(-1,self.input_size)
    for i, layer in enumerate(self.layers):
        x = layer(x)

    if self.args.task == "binary_classification":
        x = torch.sigmoid(x)
    elif self.args.task == "classification":
        x = F.softmax(x, dim=-1)
    return x, torch.tensor([0.0]).view(1).to(input.device)
  

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

        self.dropout = DROPOUT_FACTORY[self.args.dropout_type](self.args.p)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2,  bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2,  bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=50*7*7, out_features=500,  bias=False)
        self.linear2 = nn.Linear(in_features=500, out_features=output_size,  bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool2(x)
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=-1)
        return x, torch.tensor([0.0]).view(1).to(x.device)

    def log(self, *args, **kwargs):
        pass
  
