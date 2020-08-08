import sys
import torch
import argparse
from datetime import timedelta
import numpy as np
from tqdm import tqdm


sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

from experiments.plot_settings import MLP, PLT
from experiments.utils import plot_regression_uncertainty

from src.data import *
from src.trainer import Trainer
from src.models.stochastic import STOCHASTIC_FACTORY
from src.losses import LOSS_FACTORY
from src.models.stochastic.sgld.utils import SGLD
import src.utils as utils
from experiments.utils import moving_average

parser = argparse.ArgumentParser("pointwise_regression")

parser.add_argument('--task', type=str, default='regression', help='the main task; defines loss')
parser.add_argument('--model_type', type=str, default='stochastic', help='whether the model is pointwise or stochastic')
parser.add_argument('--model', type=str, default='linear_sgld', help='the model that we want to train')

parser.add_argument('--learning_rate', type=float,
                    default=0.008, help='init learning rate')
parser.add_argument('--loss_scaling', type=str,
                    default='whole', help='smoothing factor')
parser.add_argument('--smoothing', type=float,
                    default=0.0, help='smoothing factor')
parser.add_argument('--weight_decay', type=float,
                    default=.05, help='weight decay')
parser.add_argument('--clip', type=float,
                    default=0., help='dropout probability')
                    
parser.add_argument('--data', type=str, default='./../../../data/',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='regression',
                    help='dataset')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--kernel', action='store_true', default=True,
                    help='whether to use a gaussian kernel for the data')
parser.add_argument('--num_features', type=int, default=10,
                    help='number of features for the kernel')

parser.add_argument('--dataset_size', type=float,
                    default=1.0, help='portion of the whole training data')
parser.add_argument('--valid_portion', type=float,
                    default=0.1, help='portion of training data')

parser.add_argument('--burnin_epochs', type=int,
                    default=800, help='portion of training data')
parser.add_argument('--epochs', type=int, default=1000,
                    help='num of training epochs')

parser.add_argument('--layers', nargs='+',
                    default=[512, 512, 512], help='num of init channels')
parser.add_argument('--activation', type=str, default='relu', help='specify the activation')
parser.add_argument('--input_size', nargs='+',
                    default=[10], help='input size')
parser.add_argument('--output_size', type=int,
                    default=1, help='output size')
parser.add_argument('--samples', type=int,
                    default=1, help='output size')

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--save_last', action='store_true', default=True,
                    help='whether to just save the last model') 
parser.add_argument('--no_logging', action='store_true',
                    help='whether to performal logging in the models')
parser.add_argument('--model_path', type=str,
                    default='', help='path to save the model weights')

parser.add_argument('--num_workers', type=int,
                    default=0, help='number of workers')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--debug', action='store_true', help='whether we are currently debugging')

parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default = 0, help='gpu device ids')


parser.add_argument('--train', action='store_true',
                    help='whether to performal training')
parser.add_argument('--analyse', action='store_true',
                    help='whether to analyse')


def main():
  args = parser.parse_args()
  args, logging, writer = utils.parse_args(args)
  
  logging.info('# Start Re-training #')
  
  criterion = LOSS_FACTORY[args.task](args, args.loss_scaling)

  if args.model_type == "stochastic":
    model_temp = STOCHASTIC_FACTORY[args.model]
  else:
    raise NotImplementedError("Other models have not been implemented!")
  model = model_temp(args.input_size, args.output_size, args.layers, args.activation, args, True)
    
  logging.info('## Model created: ##')
  logging.info(model.__repr__())
    
  logging.info("### Param size = %f MB, Total number of params = %d ###" %
              utils.count_parameters_in_MB(model, args))

  logging.info('### Loading model to parallel GPUs ###')

  utils.profile(model, args, logging)
  model = utils.model_to_gpus(model, args)
  
  logging.info('### Preparing schedulers and optimizers ###')
  optimizer = SGLD(
      model.parameters(),
      args.learning_rate,
      norm_sigma = args.weight_decay)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, args.epochs)

  logging.info('## Downloading and preparing data ##')
  train_loader, valid_loader= get_train_loaders(args)
  

  logging.info('## Beginning Training ##')

  train = Trainer(model, criterion, optimizer, scheduler, args)

  best_error, train_time, val_time = train.train_loop(
      train_loader, valid_loader, logging, writer)

  logging.info('## Finished training, the best observed validation error: {}, total training time: {}, total validation time: {} ##'.format(
      best_error, timedelta(seconds=train_time), timedelta(seconds=val_time)))

  logging.info('## Beginning Plotting ##')
  del model 
  with torch.no_grad():
    args.samples = 100
    args.model_path = args.save
    model = model_temp(args.input_size, args.output_size, args.layers, args.activation, args, False)
    model = utils.model_to_gpus(model, args)
    model.eval()
    plot_regression_uncertainty(model, PLT, train_loader, args)
    logging.info('# Finished #')
    

if __name__ == '__main__':
  main()