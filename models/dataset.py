import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset

import os

def get_cifar10(batch_size, data_root='~/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_cifar100(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_imagenet(batch_size, data_root='/home/guangxian-z/Data', train=True, val=True, **kwargs):
    # data_root = data_root
    num_workers = kwargs.setdefault('num_workers', 1)
    print("Building ImageNet data loader with {} workers".format(num_workers))
    
    ds = []
    if train:
        transform=transforms.Compose([
            # transforms.Pad(4),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # train_path = os.path.join(data_root, 'train')
        # imagenet_traindata = datasets.ImageFolder(train_path, transform=transform)

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageNet(split='train', root=data_root, transform=transform, download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)
        ds.append(train_loader)
    if val:
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # val_path = os.path.join(data_root, 'val')
        # imagenet_testdata = datasets.ImageFolder(val_path, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageNet(split='val', root=data_root, transform=transform, download=True),
            batch_size=batch_size, 
            shuffle=False, 
            **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_mnist(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building MNIST data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(2),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Pad(2),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_mnist_sc(batch_size, data_root='/tmp/public_dataset/pytorch' ):
    # Data preparation
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Combine the datasets
    entire_dataset = ConcatDataset([train_data, test_data])
    entire_dataloader = DataLoader(entire_dataset, batch_size=batch_size, shuffle=True)

    # small dataset
    num_samples = len(test_data)  # Total number of samples in the test data
    subset_size = int(0.01 * num_samples)  # 1% of the total size
    # Randomly sample indices for the subset
    subset_indices = torch.randperm(num_samples)[:subset_size]
    # Create a subset of the original dataset
    test_data_subset = Subset(test_data, subset_indices)
    test_loader_s = DataLoader(test_data_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader_s

def get_fashionmnist(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building MNIST data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(2),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Pad(2),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_fashionmnist_sc(batch_size, data_root='/tmp/public_dataset/pytorch' ):
    # Data preparation
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Combine the datasets
    entire_dataset = ConcatDataset([train_data, test_data])
    entire_dataloader = DataLoader(entire_dataset, batch_size=batch_size, shuffle=True)

    # small dataset
    num_samples = len(test_data)  # Total number of samples in the test data
    subset_size = int(0.01 * num_samples)  # 0.5% of the total size
    # Randomly sample indices for the subset
    subset_indices = torch.randperm(num_samples)[:subset_size]
    # Create a subset of the original dataset
    test_data_subset = Subset(test_data, subset_indices)
    test_loader_s = DataLoader(test_data_subset, batch_size=1, shuffle=False)

    return train_loader, test_loader_s

