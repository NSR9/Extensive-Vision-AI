"""
contains all methods related to data
"""
import torch
from torchvision import datasets, transforms


def download_train_data(train_transforms):
    """
    Downloads Train Data
    Args:
        train_transforms: Applies transformations on train data

    Returns: train data

    """
    train_data = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=train_transforms
    )
    return train_data


def download_test_data(test_transforms):
    """
    Download Test  Data
    Args:
        test_transforms: Transformations to be applied on test data

    Returns: test data

    """
    test_data = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=test_transforms
    )
    return test_data


def load_train_data(train_data, **data_loader_args):
    """
    Load Train Data
    Args:
        train_data: train data
        **data_loader_args: additional params used while loading dataa

    Returns: train loader

    """
    train_loader = torch.utils.data.DataLoader(
        train_data,
        **data_loader_args
    )
    return train_loader


def load_test_data(test_data, **data_loader_args):
    """
    Load Test Data
    Args:
        test_data: test data
        **data_loader_args: additional params used while using loading data

    Returns: test loader

    """
    test_loader = torch.utils.data.DataLoader(
        test_data,
        **data_loader_args
    )
    return test_loader


def train_data_transformation():
    """
    Set of transformations to be applied on train data
    Returns: list of transformations

    """
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-6.0, 6.0), fill=(1,)),
        transforms.RandomAffine(degrees=7, shear=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.40, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        # Note the difference between (0.1307) and (0.1307,)
    ])
    return train_transforms


def test_data_transformation():
    """
    Set of transforms to be applied on test data
    Returns: list of transforms

    """
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return test_transforms


def get_data_loader_args(cuda):
    return dict(shuffle=True, batch_size=128, num_workers=1, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

