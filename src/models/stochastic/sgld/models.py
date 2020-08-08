import torch.nn as nn
import torch.nn.functional as F
import torch 
import os
from src.utils import load_model,  atoi, natural_keys
import re 

class _LinearNetwork(nn.Module):
  def __init__(self, input_size, output_size, layers, activation, args):
    super(_LinearNetwork, self).__init__()
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
            self.layers.append(nn.Linear(self.input_size, int(layers[0]), bias=False))
        else:
            self.layers.append(nn.Linear(int(layers[i-1]), int(layers[i]), bias=False))
        self.layers.append(self.activation())

    self.layers.append(nn.Linear(int(layers[len(layers)-1]), self.output_size, bias=False))
    
  def forward(self, input):
    x = input.view(-1,self.input_size)
    for i, layer in enumerate(self.layers):
        x = layer(x)
    if self.args.task == "binary_classification":
        x = torch.sigmoid(x)
    elif self.args.task == "classification":
        x = F.softmax(x, dim=-1)
    return x, torch.tensor([0.0]).to(x.device).view(1)
  

  def log(self, *args, **kwargs):
    pass

class LinearNetwork(nn.Module):
  def __init__(self, input_size, output_size, layers, activation, args, training_mode=True):
    super(LinearNetwork, self).__init__()
    self.args = args
    self.input_size = 1

    for i in input_size:
        self.input_size*=int(i) 
    self.output_size = int(output_size)
    if training_mode:
        self.main_net = _LinearNetwork(input_size, output_size, layers, activation, args)
    else:
        self.ensemble = nn.ModuleList([])
        self._load_ensemble(input_size, output_size, layers, activation)
        self.counter = 0
    self.training_mode = training_mode
  
  def reset_parameters(self):
      pass
  def _load_ensemble(self, input_size, output_size, layers, activation):
      sample_names = []
      for root, dirs, files in os.walk(self.args.save):
        for filename in files:
          if ".pt" in filename:
            sample_name = re.findall('weights_[0-9]*.pt', filename)
            if len(sample_name)>=1:
                sample_name = sample_name[0]
                sample_names.append(sample_name)
      sample_names.sort(key=natural_keys)
      sample_names = sample_names[:self.args.samples]  
      for i in range(self.args.samples):
        model = _LinearNetwork(input_size, output_size, layers, activation, self.args)
        load_model(model, self.args.model_path + "/" + sample_names[i])
        self.ensemble.append(model)
    
  def forward(self, input):
    if self.training_mode:
        return self.main_net(input)
    else:
        out = self.ensemble[self.counter](input)
        self.counter+=1
        if self.counter >= self.args.samples:
            self.counter = 0
        return out

  def log(self, *args, **kwargs):
    if self.training_mode:
        self.main_net.log()
    else:
        for net in self.ensemble:
            net.log()

class _ConvNetwork(nn.Module):
    def __init__(self, input_size, output_size, layers, activation, args):
        super(_ConvNetwork, self).__init__()
        self.args = args 

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "linear":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError("Other activations have not been implemented!")

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5,5), stride=1, padding=2, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5), stride=1, padding=2,  bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=50*7*7, out_features=500,  bias=False)
        self.linear2 = nn.Linear(in_features=500, out_features=output_size,  bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=-1)
        return x, torch.tensor([0.0]).to(x.device).view(1)

    def log(self, *args, **kwargs):
        pass
  
class ConvNetwork(nn.Module):
  def __init__(self, input_size, output_size, layers, activation, args, training_mode=True):
    super(ConvNetwork, self).__init__()
    self.args = args
    self.input_size = 1

    for i in input_size:
        self.input_size*=int(i) 
    self.output_size = int(output_size)
    if training_mode:
        self.main_net = _ConvNetwork(input_size, output_size, layers, activation, args)
    else:
        self.ensemble = nn.ModuleList([])
        self._load_ensemble(input_size, output_size, layers, activation)
        self.counter = 0
    self.training_mode = training_mode
  def _load_ensemble(self, input_size, output_size, layers, activation):
      sample_names = []
      for root, dirs, files in os.walk(self.args.save):
        for filename in files:
          if ".pt" in filename:
            sample_name = re.findall('weights_[0-9]*.pt', filename)
            if len(sample_name)>=1:
                sample_name = sample_name[0]
                sample_names.append(sample_name)
      sample_names.sort(key=natural_keys)
      sample_names = sample_names[:self.args.samples]
      for i in range(self.args.samples):
          model = _ConvNetwork(input_size, output_size, layers, activation, self.args)
          load_model(model, self.args.model_path + "/" + sample_names[i])
          self.ensemble.append(model)
    
  def forward(self, input):
    if self.training_mode:
        return self.main_net(input)
    else:
        out = self.ensemble[self.counter](input)
        self.counter+=1
        if self.counter == len(self.ensemble):
            self.counter = 0
        return out

  def log(self, *args, **kwargs):
    if self.training_mode:
        self.main_net.log()
    else:
        for net in self.ensemble:
            net.log()