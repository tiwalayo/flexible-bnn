import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import scale
import random
import os

def regression_function(X):
    return np.cos(10*X**2) + 0.1 * np.sin(100 * X)

def gaussian(_x, K, scale=0.15):
    mus = np.reshape(np.linspace(0, 1, K), (K, 1))
    x = np.empty(shape=(len(_x),K))
    for i in range(len(_x)):
        x[i][0] = _x[i]
        for j in range(1, K):
            x[i][j] = np.exp(-(_x[i]-mus[j-1])**2/(2*scale**2))
    return x

def regression_data_generator(a=-0.1, b=0.9, N_points=1000, args=None, X = None):
    if X is None:
        X = np.reshape(np.random.uniform(low=a, high = b,size=N_points), (N_points, 1))
    Y = regression_function(X)
    if args.kernel:
        X_kernel = gaussian(X, args.num_features)
    else:
        X_kernel = X

    return X_kernel, Y, X


def classification_binary_data_generator(a_mean=[3.0, 3.0], b_mean=[-3,-3], cov=[[1.0, 0.0],[0.0, 1.0]], N_points=50, args=None):
    assert N_points // 2 >= 0
    X_positive = np.random.multivariate_normal(a_mean, cov, N_points//2)
    X_negative = np.random.multivariate_normal(b_mean, cov, N_points//2)
    Y_positive = np.ones((len(X_positive),1))
    Y_negative = np.zeros((len(X_negative),1))

    X = np.concatenate([X_positive, X_negative])
    Y = np.concatenate([Y_positive, Y_negative])

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    return X, Y



def get_train_loaders(args):
    assert args.dataset_size > 0 and args.dataset_size <= 1.
    assert args.valid_portion >= 0 and args.valid_portion < 1.
    train_data = None
    if args.dataset == "mnist":
        train_transform = transforms.Compose([transforms.ToTensor(), \
                                transforms.Normalize((0,), (1,))])

        train_data = datasets.MNIST(root=args.data, train=True, 
                                    download=True, transform=train_transform)
    elif args.dataset == "random":
    
        t = [transforms.ToTensor()]
        train_transform = transforms.Compose(t)

        train_data = datasets.FakeData(size=10000, image_size=args.input_size, num_classes=args.num_classes, 
                                       transform=t, random_offset=args.seed)
    elif args.dataset == "regression":
        X, Y, _ = regression_data_generator(args=args)
        inps = torch.from_numpy(X).float()
        tgts = torch.from_numpy(Y).float()
        train_data = torch.utils.data.TensorDataset(inps, tgts)

    elif args.dataset == "binary_classification":
        X, Y = classification_binary_data_generator(args=args)
        inps = torch.from_numpy(X).float()
        tgts = torch.from_numpy(Y).float()
        train_data = torch.utils.data.TensorDataset(inps, tgts) 
    else:
        raise NotImplementedError("Other datasets not implemented")
    return get_train_split_loaders(args.dataset_size, args.valid_portion, train_data, args.batch_size, args.data, args.num_workers)



def get_train_split_loaders(dataset_size, valid_portion, train_data, batch_size, path_to_save_data, num_workers=0):
    num_train = int(np.floor(len(train_data) * dataset_size))
    indices = list(range(len(train_data)))
    indices = random.sample(indices, num_train)
    valid_split = int(
        np.floor((valid_portion) * num_train))   # 40k

    valid_idx, weights_idx = indices[:valid_split], indices[valid_split:]

    weights_sampler = SubsetRandomSampler(weights_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=weights_sampler,
        pin_memory=True, num_workers=num_workers)
    
        
    valid_loader = None
    if valid_portion>0.0:
        valid_sampler = SubsetRandomSampler(valid_idx)

        valid_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=valid_sampler, 
            pin_memory=True, num_workers=num_workers)

    return train_loader, valid_loader


def get_test_loader(args):
    test_data = None
    if args.dataset == "mnist":
        test_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0,), (1,))])

        test_data = datasets.MNIST(root=args.data, train=False,
                                   download=True, transform=test_transform)
    elif args.dataset == "regression":
        X, Y, _ = regression_data_generator(a = -1, b = 2, N_points = 100, args=args)
        inps = torch.from_numpy(X).float()
        tgts = torch.from_numpy(Y).float()
        test_data = torch.utils.data.TensorDataset(inps, tgts)

    elif args.dataset == "binary_classification":
        X, Y = classification_binary_data_generator(args=args, cov = [[2.0, 0.0], [0.0, 2.0]])
        inps = torch.from_numpy(X).float()
        tgts = torch.from_numpy(Y).float()
        test_data = torch.utils.data.TensorDataset(inps, tgts)
    elif args.dataset == "random":
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])        
        test_data = datasets.FakeData(size=10000, image_size=args.input_size[1:], num_classes=10, random_offset=args.seed,transform=test_transform)
    else:
        raise NotImplementedError("Other datasets not implemented")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=args.num_workers)
    return test_loader