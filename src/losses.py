import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch
import torch.nn as nn
import math

LOSS_FACTORY = {'classification': lambda args, scaling: ClassificationLoss(args, scaling),
                'binary_classification': lambda args, scaling: BinaryClassificationLoss(args, scaling),
                'regression': lambda args, scaling: RegressionLoss(args, scaling)}

class Loss(nn.Module):
    def __init__(self, args, scaling):
        super(Loss, self).__init__()
        self.args = args
        self.scaling = scaling 

    def kl_divergence(self, kl):
        if len(kl)>1:
            return kl.mean()
        else:
            return kl

class BinaryClassificationLoss(Loss):
  def __init__(self, args, scaling):
    super(BinaryClassificationLoss, self).__init__(args, scaling)
    self.bce = nn.BCELoss()

  def forward(self, outs, targets, model, kl, gamma, n_batches, n_points):
    if self.scaling=='whole':
      bce = n_points*self.bce(outs, targets)
      kl = self.kl_divergence(kl) / n_batches
    elif self.scaling =='batch':
      bce = self.bce(outs, targets)
      kl = self.kl_divergence(kl) / (outs.shape[0]*n_batches)
    else:
      raise NotImplementedError('Other scaling not implemented!')
    loss = bce + gamma*kl
    return loss, bce, kl


class ClassificationLoss(Loss):
  def __init__(self, args, scaling):
      super(ClassificationLoss, self).__init__(args, scaling)
      self.ce = _SmoothCrossEntropyLoss(smoothing=self.args.smoothing)
  def forward(self, outs, targets, model, kl, gamma, n_batches, n_points):
    if self.scaling=='whole':
      ce = n_points*self.ce(outs, targets)
      kl = self.kl_divergence(kl) / n_batches
    elif self.scaling=='batch':
      ce = self.ce(outs, targets)
      kl = self.kl_divergence(kl) / (outs.shape[0]*n_batches)
    else:
      raise NotImplementedError('Other scaling not implemented!')
    loss = ce + gamma * kl

    return loss, ce, kl

class RegressionLoss(Loss):
  def __init__(self, args, scaling):
    super(RegressionLoss, self).__init__(args, scaling)
    self.mse = nn.MSELoss()

  def forward(self, outs, targets, model, kl, gamma, n_batches, n_points):
    if self.scaling == 'whole':
      mse = n_points*self.mse(outs, targets)
      kl = self.kl_divergence(kl) / n_batches
    elif self.scaling == 'batch':
      mse = self.mse(outs, targets)
      kl = self.kl_divergence(kl) / (outs.shape[0]*n_batches)
    else:
      raise NotImplementedError('Other scaling not implemented!')
    loss = mse + gamma*kl
    return loss, mse, kl

class _SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets, n_classes, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets, smoothing = None):
        if smoothing is None:
          smoothing = self.smoothing
        targets = _SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         smoothing)
        lsm = torch.log(inputs)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss