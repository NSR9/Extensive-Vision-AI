import torch

from models.model import Net
from utils.transform import Transforms
from utils.misc import get_cuda, run_epochs
from utils.graphs_utility import plot_graphs
from utils.data_utility import train_data_transformation, test_data_transformation, \
    download_train_data, download_test_data, get_data_loader_args, load_test_data, load_train_data


def run():
    """"
        contains all steps for training and testing model
    """
    train_transforms = Transforms(train_data_transformation())  # Train Transforms
    test_transforms = Transforms(test_data_transformation())  # Test Transforms

    train_data = download_train_data(train_transforms=train_transforms)  # Download Train Data
    test_data = download_test_data(test_transforms=test_transforms)  # Download Test Data

    cuda = get_cuda()  # Check for cuda
    data_loader_args = get_data_loader_args(cuda)  # Data Loader Arguments

    train_loader = load_train_data(train_data, **data_loader_args)  # Load Train Data
    test_loader = load_test_data(test_data, **data_loader_args)  # Load Test Data

    device = torch.device("cuda" if cuda else "cpu")

    model = Net().to(device)

    # Run Epochs
    train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values = run_epochs(
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model=model
    )

    plot_graphs(
        train_losses=train_loss_values,
        test_losses=test_loss_values,
        train_accuracy=train_accuracy_values,
        test_accuracy=test_accuracy_values
    )  # Values for plotting graphs


if __name__ == "__main__":
    run()
