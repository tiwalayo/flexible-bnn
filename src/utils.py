import os
import numpy as np
import torch
import shutil
import random
import pickle
import torch.nn.functional as F
import sys
import time
import glob
import logging
import math
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torchprof.profile import Profile
import copy
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(-?\d+)', text)]

class AverageMeter(object):

  def __init__(self):
      self.reset()

  def reset(self):
      self.avg = 0.0
      self.sum = 0.0
      self.cnt = 0.0

  def update(self, val, n=1):
      self.sum += val * n
      self.cnt += n
      self.avg = self.sum / self.cnt

class LinearScheduler():
      def __init__(self, start_value, end_value, start_epoch, end_epoch, epochs):
        self.vals = [start_value] * start_epoch + list(np.linspace(
            start_value, end_value, end_epoch-start_epoch)) + [end_value]*(epochs - end_epoch)
      
      def __getitem__(self, index):
            return self.vals[index]
    
def save_model(model, args, special_info=""):
  torch.save(model.state_dict(), os.path.join(args.save, 'weights'+special_info+'.pt'))
  torch.save(model, os.path.join(args.save, 'model'+special_info+'.pt'))
    
  with open(os.path.join(args.save, 'args.pt'), 'wb') as handle:
    pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

def profile(model, args, logging):
  x = torch.randn(args.input_size, requires_grad=False)
  model_copy = copy.deepcopy(model)
  if args.gpu!=-1:
    model_copy = model_copy.cuda()

  if next(model_copy.parameters()).is_cuda: 
    x = x.cuda()

  with torch.no_grad():
    if next(model_copy.parameters()).is_cuda: 
      logging.info("## Profiling report on a GPU ##") 
      with Profile(model_copy, use_cuda =True) as prof:
        model_copy(x)
      logging.info(prof) 

    logging.info("## Profiling report on a CPU ##") 
    cpu_model = model_copy.cpu()
    with Profile(cpu_model, use_cuda =False)  as prof:
        cpu_model(x.cpu())
    logging.info(prof)
    del cpu_model
    del model_copy

def ece(output, target):
    _ece = 0
    confidences, predictions = torch.max(output, 1)
    accuracies = predictions.eq(target)
        
    bin_boundaries = torch.linspace(0, 1, 10 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
      in_bin = confidences.gt(bin_lower.item()) * \
          confidences.le(bin_upper.item())
      prop_in_bin = in_bin.float().mean()
      if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        _ece += torch.abs(avg_confidence_in_bin -
                            accuracy_in_bin) * prop_in_bin
    return _ece

def error(output, input, target, model, args):
  with torch.no_grad():
    batch_size = target.size(0)
    if args.samples is not None and args.samples>1 and model.training is False:
      y = [output]
      for i in range(1, args.samples):
        out, _ = model(input)
        y.append(out)
      output = torch.stack(y, dim=1).mean(dim=1)

    if "classification" in args.task:
      _ece = 0
      if "binary" in args.task:
        output[output>=0.5] = 1
        output[output<0.5] =0.0
        pred = output.t()
      else:
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        _ece = ece(output, target)

      correct = pred.eq(target.view(1, -1).expand_as(pred))
      correct_k = correct[:1].view(-1).float().sum(0)
      res = 100-correct_k.mul_(100.0/batch_size)

      return res, _ece
    elif "regression" in args.task:
      return torch.sqrt(((output-target)**2).mean()), 0.0
  

def count_parameters_in_MB(model, args):
  s = 0
  for name, v in model.named_parameters():
      s += v.numel()
  return s/1e6, s

def load_pickle(path):
    file = open(path, 'rb')
    return pickle.load(file)

def load_model(model, model_path):
  if model is None:
      return torch.load(model_path, map_location=torch.device('cpu'))
  state_dict = torch.load(model_path, map_location=torch.device('cpu'))
  model_dict = model.state_dict()
  pretrained_dict = {k.replace('module.','').replace('main_net.',''): v for k,
                      v in state_dict.items()}
  pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict.keys()}
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)
    
def create_exp_dir(args, scripts_to_save=None):
  if os.path.exists(args.save):
    counter = 0
    while os.path.exists(args.save+"_"+str(counter)):
      counter+=1
    args.save = args.save+"_"+str(counter)
    
  os.mkdir(args.save)

  if scripts_to_save is not None:
    os.mkdir(os.path.join(args.save, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(args.save, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def check_tensor_in_lists(atensor, *lists):
    for l in lists:
        if (any([(atensor == t_).all() for t_ in l if atensor.shape == t_.shape])):
            return True
    return False

def model_to_gpus(model, args):
  if args.gpu!= -1:
    model = model.cuda()
  return model

def parse_args(args):
  args.save = '{}-{}-{}'.format(args.dataset, args.task, time.strftime("%Y%m%d-%H%M%S"))
    
  create_exp_dir(
      args, scripts_to_save=glob.glob('*.py') + glob.glob('*.sh') + glob.glob('../../src/**/*.py', recursive=True) + glob.glob('../*.py'))
  
    
  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.log'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)
  
  print('Experiment dir : {}'.format(args.save))

  writer = SummaryWriter(
      log_dir=args.save+"/",max_queue=5)
  
  if torch.cuda.is_available() and args.gpu!=-1:
    logging.info('## GPUs available = {} ##'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
  else:
    logging.info('## No GPUs detected ##')
    
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  logging.info("## Args = %s ##", args)
    
  return args, logging, writer

      

          
  
