
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import src.utils as utils
import os
import numpy as np
import time 

class Trainer():
  def __init__(self, model, criterion, optimizer, scheduler, args):
    super().__init__()
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.args = args
    
    self.train_step = 0
    self.train_time = 0.0
    self.val_step = 0
    self.val_time = 0.0

    if hasattr(args, 'gamma_start_epoch'):
      self.gamma_scheduler = utils.LinearScheduler(args.gamma_start_value, args.gamma_end_value, args.gamma_start_epoch, args.gamma_end_epoch, args.epochs)
    else:
      self.gamma_scheduler = utils.LinearScheduler(0.0, 0.0, 0, args.epochs, args.epochs)
    
  def _scalar_logging(self, obj, main_obj, kl, error_metric, ece, info, iteration, writer):
    if "classification" in self.args.task:
        _error_metric, _main_obj = 'error', 'ce'
    elif self.args.task == "regression":
        _error_metric, _main_obj = 'rmse', 'rmse'
    writer.add_scalar(info+_error_metric, error_metric, iteration)
    writer.add_scalar(info+'loss', obj, iteration)
    writer.add_scalar(info+_main_obj, main_obj, iteration)
    writer.add_scalar(info+'kl', kl, iteration)
    writer.add_scalar(info+'ece', ece, iteration)
    
  def _get_average_meters(self):
    error_metric = utils.AverageMeter()
    obj = utils.AverageMeter()
    main_obj = utils.AverageMeter()
    kl = utils.AverageMeter()
    ece = utils.AverageMeter()
    return error_metric, obj, main_obj, kl, ece
    
  def train_loop(self, train_loader, valid_loader, logging, writer=None):
    best_error = float('inf')
    train_error_metric = train_obj = train_main_obj = train_ece = train_kl = None
    
    for epoch in range(self.args.epochs):
      if epoch >= 1 and self.scheduler is not None:
        self.scheduler.step()
      
      if self.scheduler is not None:
        lr = self.scheduler.get_last_lr()[0]
      else:
        lr = self.args.learning_rate

      if writer is not None:
        writer.add_scalar('Train/learning_rate', lr, epoch)
        writer.add_scalar('Train/gamma', self.gamma_scheduler[epoch], epoch)
      logging.info(
          '### Epoch: [%d/%d], Learning rate: %e, Gamma: %e ###', self.args.epochs,
          epoch, lr, self.gamma_scheduler[epoch])
      
   
      train_obj, train_main_obj, train_kl, train_error_metric, train_ece = self.train(epoch, train_loader, self.optimizer, logging, writer)
      
      logging.info('#### Train | Error: %f, Train loss: %f, Train main objective: %f, Train KL: %f, Train ECE %f ####',
                     train_error_metric, train_obj, train_main_obj, train_kl, train_ece)

      
      if writer is not None:
        self._scalar_logging(train_obj, train_main_obj, train_kl, train_error_metric, train_ece, "Train/", epoch, writer)
    

      # validation
      val_obj, val_main_obj, val_kl, val_error_metric, val_ece = self.infer(epoch,
                                                        valid_loader, logging, writer, "Valid")
      logging.info('#### Valid | Error: %f, Valid loss: %f, Valid main objective: %f, Valid KL: %f, Valid ECE %f ####',
                    val_error_metric, val_obj, val_main_obj, val_kl, val_ece)
        

      if writer is not None:
        self._scalar_logging(val_obj, val_main_obj, val_kl, val_error_metric, val_ece, "Valid/", epoch, writer)
      
      if val_error_metric <= best_error or self.args.save_last:
        special_infor = ""
        # Avoid correlation between the samples
        if hasattr(self.args, 'burnin_epochs') and epoch>=self.args.burnin_epochs and epoch%2==0:
          special_infor = "_"+str(epoch)
        utils.save_model(self.model, self.args, special_infor)
        best_error = val_error_metric
        logging.info(
            '### Epoch: [%d/%d], Saving model! Current best error: %f ###', self.args.epochs,
            epoch, best_error)
      
    return best_error, self.train_time, self.val_time
  
  def _step(self, input, target, optimizer, epoch, n_batches, n_points, train_timer):
    start = time.time()

    input = Variable(input, requires_grad=False)
    target = Variable(target, requires_grad=False)
    if next(self.model.parameters()).is_cuda:
      input = input.cuda()
      target = target.cuda()
      
    if optimizer is not None:
      optimizer.zero_grad()
    output, kl = self.model(input)
    obj, main_obj, kl = self.criterion(output, target, self.model, kl, self.gamma_scheduler[epoch], n_batches, n_points)
    
    if optimizer is not None:
      obj.backward()
      if self.args.clip>0:
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip)
      optimizer.step()
      
    error_metric, ece = utils.error(output, input, target, self.model, self.args)

    if train_timer:
      self.train_time += time.time() - start
    else:
      self.val_time += time.time() - start
      
    return error_metric.item(), obj.item(), main_obj.item(), kl.item(), ece


  def train(self, epoch, loader, optimizer, logging, writer):
    error_metric, obj, main_obj, kl, ece = self._get_average_meters()
    self.model.train()
    
    for step, (input, target) in enumerate(loader):
      n = input.shape[0]
      _error_metric, _obj, _main_obj, _kl, _ece= self._step(input, target, optimizer, epoch, len(loader), n*len(loader), True)
      
      obj.update(_obj, n)
      main_obj.update(_main_obj, n)
      kl.update(_kl, n)
      error_metric.update(_error_metric, n)
      ece.update(_ece, n)

      if step % self.args.report_freq == 0:
        logging.info('##### Train step: [%03d/%03d] | Error: %f, Loss: %f, Main objective: %f, KL: %f, ECE: %f #####',
                       len(loader),  step, error_metric.avg, obj.avg, main_obj.avg, kl.avg, ece.avg)
        if writer is not None:
          self._scalar_logging(obj.avg, main_obj.avg, kl.avg, error_metric.avg, ece.avg, 'Train/Iteration/', self.train_step, writer)
          if not self.args.no_logging:
              self.model.log(writer, self.train_step, 'Train/')

        self.train_step += 1
        
      if self.args.debug:
        break
    
    return obj.avg, main_obj.avg, kl.avg, error_metric.avg, ece.avg

  def infer(self, epoch, loader, logging, writer, dataset="Valid"):
    with torch.no_grad():
      error_metric, obj, main_obj, kl, ece = self._get_average_meters()
      self.model.eval()

      for step, (input, target) in enumerate(loader):
        n = input.shape[0]
        _error_metric, _obj, _main_obj, _kl, _ece = self._step(
             input, target, None, epoch, len(loader), n * len(loader), False)

        obj.update(_obj, n)
        main_obj.update(_main_obj, n)
        kl.update(_kl, n)
        error_metric.update(_error_metric, n)
        ece.update(_ece, n)
        
        if step % self.args.report_freq == 0:
          logging.info('##### {} step: [{}/{}] |  Error: {}, Loss: {}, Main objective: {}, KL: {}, ECE: {} #####'.format(
                       dataset, len(loader), step, error_metric.avg, obj.avg, main_obj.avg, kl.avg, ece.avg))
          if writer is not None:
            self._scalar_logging(obj.avg, main_obj.avg, kl.avg, error_metric.avg, ece.avg, '{}/Iteration/'.format(dataset), self.val_step, writer)

            self.val_step += 1
          
        if self.args.debug:
          break

      return obj.avg, main_obj.avg, kl.avg, error_metric.avg, ece.avg