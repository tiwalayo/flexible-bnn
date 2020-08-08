import torch 
from torch.optim import Optimizer
from torch.distributions import Normal
import numpy as np

class SGLD(Optimizer):
    def __init__(self, params, lr, norm_sigma=0.0, alpha=0.99, eps=1e-8, centered=False, addnoise=True, p=True):
        weight_decay = 1/(norm_sigma ** 2 + eps)
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, alpha=alpha, weight_decay=weight_decay, eps=eps, centered=centered, addnoise=addnoise, p=p)
        super(SGLD, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    d_p.add_(p.data, alpha=group['weight_decay'])
                
                
                square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1-alpha)
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(d_p, alpha=1-alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                    
                
                if group['addnoise']:
                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(group['lr'])
                    if group['p']:
                        p.data.add_(0.5 * d_p.div_(avg) + langevin_noise / torch.sqrt(avg), alpha=-group['lr'])
                    else:
                        p.data.add_(0.5 * d_p + langevin_noise, alpha=-group['lr'])
                else:
                    if group['p']:
                        p.data.addcdiv_( 0.5 * d_p, avg, value = -group['lr'])
                    else:
                        p.data.addcdiv_( 0.5 * d_p, value = -group['lr'])

        return loss