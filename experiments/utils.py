import numpy as np
import torch
from scipy.ndimage.interpolation import rotate
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import multivariate_normal

import sys

sys.path.append('../')
from src.data import *


def plot_rotated_digit(x, path, plt):
    X1 = np.array([rotate(x, i, reshape=False) for i in range(50, 130, 7)])
    X1 = X1.reshape(X1.shape[0], 1, X1.shape[1], X1.shape[2])
    plt.figure(figsize=(8, 1))

    gs = gridspec.GridSpec(1, 12)
    gs.update(wspace=0, hspace=0)

    for i in range(len(X1)):
        plt.subplot(gs[i])
        plt.imshow(X1.squeeze()[i], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(path+'/rotated_one.png')
    return X1

def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

def plot_mnist_uncertainty(model, plt, test_loader, args):
    outputs = []
    targets = []

    for i, (input, target) in enumerate(test_loader):
      input = torch.autograd.Variable(input, requires_grad=False)
      target = torch.autograd.Variable(target, requires_grad=False)
      if next(model.parameters()).is_cuda:
        input = input.cuda()
        target = target.cuda()
      samples = []
      for j in range(args.samples):
        logits, _ = model(input)
        logits = logits.detach()
        samples.append(logits)
      outputs.append(torch.stack(samples,dim=1).mean(dim=1))
      targets.append(target)
    outputs = torch.cat(outputs, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()
    f = _plot_ece(outputs, targets, plt)
    plt.savefig(args.save+'/ece_test.png')
    f = _plot_model_certainty(outputs, plt)
    plt.savefig(args.save+'/certainty_test.png')


    args.dataset='random'
    test_loader = get_test_loader(args)
    outputs = []
    targets = []
    for i, (input, target) in enumerate(test_loader):
      input = torch.autograd.Variable(input, requires_grad=False)
      target = torch.autograd.Variable(target, requires_grad=False)
      if next(model.parameters()).is_cuda:
        input = input.cuda()
        target = target.cuda()
      samples=[]
      for j in range(args.samples):
        logits, _ = model(input)
        logits = logits.detach()
        samples.append(logits)
      outputs.append(torch.stack(samples,dim=1).mean(dim=1))
      targets.append(target)
    outputs = torch.cat(outputs, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()
    f = _plot_ece(outputs, targets, plt)
    plt.savefig(args.save+'/ece_random.png')
    f = _plot_model_certainty(outputs, plt)
    plt.savefig(args.save+'/certainty_random.png')

    args.dataset='mnist'
    args.samples=100
    test_loader = get_test_loader(args)
    X_test, _ = iter(test_loader).next()
    digit = X_test[2].numpy().squeeze()  
    rotated_digits = plot_rotated_digit(digit, args.save, plt)
    rotated_digits = torch.tensor(rotated_digits)

    _y = []
    for i in range(args.samples):
      logits, _ = model(rotated_digits.cuda())
      logits = logits.detach().cpu()
      _y.append(logits.numpy())

    logits = np.array(_y)
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange','deeppink', 'aqua']
    r = np.arange(1, logits.shape[1]+1)
    alpha = 1. if logits.shape[0] == 1 else 0.5
    for i in range(logits.shape[-1]):
        plt.scatter(np.tile(r, logits.shape[0]), logits[:, :, i].flatten(), \
                color=colors[i], marker='_', linewidth=None, alpha=alpha, label=str(i))

    plt.scatter(np.tile(r, 1), np.mean(logits[:, :, 1],axis=0).flatten(), \
                color=colors[1], marker='o', s=10, linewidth=10,  alpha=1.0)
    plt.title('Softmax output scatter')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(args.save+'/rotated_digit_uncertainty.png')

def plot_regression_uncertainty(model, plt, train_loader, args):
    N_points = 25

    assert N_points <= args.batch_size

    train_points, _ = next(iter(train_loader))

    train_points = train_points[:N_points]

    train_points_plt= train_points[:, 0].numpy()
    
    fig, ax = plt.subplots(1, 1)

    # Create grid values first.
    N_GRID = 1000
    xi = np.linspace(-0.4, 1.2, N_GRID)
    yi = regression_function(xi)

    xi_transformed = torch.tensor(gaussian(xi, args.num_features)).float().cuda()
    batch_size = 25

    _y_mean = []
    _y_std = []
    for i in tqdm(range(0, len(train_points)//batch_size)):
      _y = []
      for j in range(args.samples):
        sample, _ = model(train_points[i*batch_size:(1+i)*batch_size].cuda())
        _y.append(sample.detach().cpu())
      _y_mean.append(torch.stack(_y, dim=1).mean(dim=1))
      _y_std.append(torch.stack(_y, dim=1).std(dim=1))

    y_train_mean = torch.cat(_y_mean, dim=0).flatten().numpy()
    y_train_std =torch.cat(_y_std, dim=0).flatten().numpy()

    _y_mean = []
    _y_std = []
    for i in tqdm(range(0, len(xi_transformed)//batch_size)):
      _y = []
      for j in range(args.samples):
        sample, _ = model(xi_transformed[i*batch_size:(1+i)*batch_size].cuda())
        _y.append(sample.detach().cpu())
      _y_mean.append(torch.stack(_y, dim=1).mean(dim=1))
      _y_std.append(torch.stack(_y, dim=1).std(dim=1))

    y_complete_mean = torch.cat(_y_mean, dim=0).flatten().numpy()
    y_complete_std = torch.cat(_y_std, dim=0).flatten().numpy()

    ax.plot(xi, yi, label='True function', color='k')
    ax.plot(xi, y_complete_mean, label='Predicted function', color='r', linestyle='--')
    if args.samples>1:
        ax.fill_between(xi, y_complete_mean-2*y_complete_std, y_complete_mean+2*y_complete_std, color='r', alpha=0.5)

    ax.scatter(train_points_plt, y_train_mean, label="Train samples", color='g')
    ax.errorbar(train_points_plt, y_train_mean, yerr=2*y_train_std, color='g', fmt='o')
    
    ax.axvline(x=-0.1, ymin=-1, ymax=1, linestyle='-', color='g', alpha=0.2)
    ax.axvline(x=0.9, ymin=-1, ymax=1, linestyle='-', color='g', alpha=0.2)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(args.save+'/regression.png')

def plot_binary_uncertainty(model, plt, train_loader, args):

    N_points = 50
    assert N_points <= args.batch_size
    train_points, _ = next(iter(train_loader))
    train_points= train_points[:N_points].numpy()
    fig, ax = plt.subplots(1, 1)

    N_GRID = 1000
    # Create grid values first.
    xi = np.linspace(-7.5, 7.5, N_GRID)
    yi = np.linspace(-7.5, 7.5, N_GRID)

    z = torch.cartesian_prod(torch.tensor(xi).float(), torch.tensor(yi).float())
    batch_size = 1000
    _z = []
    for i in tqdm(range(0,len(z)//batch_size)):
      _y = []
      for j in range(args.samples):
        logits, _ = model(z[i*batch_size:(1+i)*batch_size].cuda())
        _y.append(logits.detach().cpu())

      _y = torch.stack(_y, dim=1).mean(dim=1)
      _z.append(_y)

    z = torch.cat(_z, dim=0).numpy()
    z = z.reshape(N_GRID, N_GRID)

    xi,yi = np.meshgrid(xi, yi)

    cp = ax.contourf(xi,yi, z)
    cbar = fig.colorbar(cp)
    cbar.set_label('Sigmoid output', rotation=270, labelpad=15)

    c = ax.contour(xi, yi, z,levels = [0.5],
                colors=('w',),linestyles=('-',),linewidths=(2,))
    ax.clabel(c, inline=1, fontsize=10)
    
    ax.scatter(train_points[:,0], train_points[:,1], label="Train samples", color='g')

    x_pos = np.linspace(-3, 7.5, 1000)
    y_pos = np.linspace(-3, 7.5, 1000)
    pos = np.transpose([np.tile(x_pos, len(y_pos)), np.repeat(y_pos, len(x_pos))])
    x_pos,y_pos = np.meshgrid(x_pos, y_pos)

    x_neg = np.linspace(-7.5, 3, 1000)
    y_neg = np.linspace(-7.5, 3, 1000)
    neg = np.transpose([np.tile(x_neg, len(y_neg)), np.repeat(y_neg, len(x_neg))])
    x_neg,y_neg = np.meshgrid(x_neg, y_neg)

    z_pos = multivariate_normal.pdf(pos, mean=[3.0, 3.0], cov=[[1.0, 0.0],[0.0, 1.0]]).reshape(1000, 1000)
    z_neg = multivariate_normal.pdf(neg, mean=[-3.0, -3.0], cov=[[1.0, 0.0],[0.0, 1.0]]).reshape(1000, 1000)
    ax.contour(x_pos, y_pos, z_pos, levels = 4,
                colors=('k',),linestyles=('--',),linewidths=(1,))
    ax.contour(x_neg, y_neg, z_neg, levels = 4,
                colors=('k',),linestyles=('--',),linewidths=(1,))

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(args.save+'/probability.png')


def _plot_ece(outputs, labels, plt, n_bins=10):
    confidences, predictions = outputs.max(1)
    accuracies = torch.eq(predictions, labels)
    f, rel_ax = plt.subplots(1, 1, figsize=(4, 2.5))

    bins = torch.linspace(0, 1, n_bins + 1)

    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    bin_corrects = np.nan_to_num(np.array([torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices]))
    bin_scores = np.nan_to_num(np.array([torch.mean(confidences[bin_index]) for bin_index in bin_indices]))
  
    confs = rel_ax.bar(bins[:-1], np.array(bin_corrects), align='edge', width=width, alpha=0.75, edgecolor='b')
    gaps = rel_ax.bar(bins[:-1], bin_scores -bin_corrects, align='edge',
                      bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    rel_ax.plot([0, 1], [0, 1], '--', color='gray')
    rel_ax.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='small')
    rel_ax.set_ylabel('Accuracy')
    rel_ax.set_xlabel('Confidence')
    plt.tight_layout()
    return f

def _plot_model_certainty(outputs, plt, n_bins=10):
    confidences, _ = outputs.max(1)
    confidences = np.nan_to_num(confidences)
    f, rel_ax = plt.subplots(1, 1, figsize=(4, 2.5))
    bin_height,bin_boundary = np.histogram(confidences,bins=n_bins)
    width = bin_boundary[1]-bin_boundary[0]
    bin_height = bin_height/float(max(bin_height))
    rel_ax.bar(bin_boundary[:-1],bin_height,width = width, align='center', color='b', label="Normalized counts")
    rel_ax.legend()
    rel_ax.set_xlabel('Confidence')
    f.tight_layout()

    return f