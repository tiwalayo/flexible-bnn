import torch 
import numpy as np
import torch.nn.functional as F

def kl_divergence(mu, sigma, mu_prior, sigma_prior):
    kl = 0.5 * (2 * torch.log(sigma_prior / sigma) - 1 + (sigma / sigma_prior).pow(2) + ((mu_prior - mu) / sigma_prior).pow(2)).sum() 
    return kl

def normpdf(x, mu=0.0, sigma=0.3):
    m = torch.distributions.Normal(torch.tensor([mu]).to(x.device), torch.tensor([sigma]).to(x.device))
    return torch.exp(m.log_prob(x))

def KumaraswamyKL(A, B, prior=None, n_samples=100):
    GAMMA = 0.57721566490153286060651209008240243104215933593992
    return -((1-1/B) + (1-1/A) * (GAMMA + torch.log(B)) - torch.log(A*B)).sum()
    
    if not prior:
        raise ValueError("You need to supply a prior.")
    eps = 1e-20
    T_ = lambda x, a, b: 2*(torch.pow(1 - torch.pow(1-x,1/b), 1/a))-1
    Kpdf = lambda x, a, b: a * b * torch.pow((x+1)/2,a-1) * torch.pow((1-torch.pow((x+1)/2,a)), b-1)
    
    def logratio(x):
        noise = torch.FloatTensor(n_samples).uniform_(0, 1).to(x.device)
        samples = T_(noise, x[0], x[1])
        return torch.log(eps+Kpdf(samples, x[0], x[1])) - torch.log(eps + prior(samples))
    
    params = torch.unbind(torch.cat((A.unsqueeze(0),B.unsqueeze(0)),dim=0).view(2,-1),dim=1)
    s =torch.cat([logratio(p) for p in params]).sum()
    return s