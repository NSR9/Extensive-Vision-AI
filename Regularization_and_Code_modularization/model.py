import torch

from utils.data_utility import download_train_data, download_test_data, \
    train_data_transformation, test_data_transformation, \
    load_train_data, load_test_data, get_data_loader_args
from utils.misc import get_cuda, run_epochs
from utils.graphs_utility import plot_accuracy_curves, wrong_predictions

from models.batch_norm import BatchNormModel
from models.group_norm import GroupNormModel
from models.layer_norm import LayerNormModel

# Using unique particular model will be loaded
MODELS = {
    "GroupNorm": GroupNormModel(),
    "LayerNorm": LayerNormModel(),
    "BatchNorm": BatchNormModel()
}


def run():
    """"
        contains all steps for training and testing model
    """
    train_transforms = train_data_transformation()  # Train Transforms
    test_transforms = test_data_transformation()  # Test Transforms

    train_data = download_train_data(train_transforms=train_transforms)  # Download Train Data
    test_data = download_test_data(test_transforms=test_transforms)  # Download Test Data

    cuda = get_cuda()  # Check for cuda
    data_loader_args = get_data_loader_args(cuda)  # Data Loader Arguments

    train_loader = load_train_data(train_data, **data_loader_args)  # Load Train Data
    test_loader = load_test_data(test_data, **data_loader_args)  # Load Test Data

    """
        models dict explanation
        key - unique to get model params 
        value - has three values
            value1 unique key which is used to get model instance
            value2 is l1 norm required or not
            value3 is l2 norm required or not
    """
    models = {
        "group_norm": ["GroupNorm", False, False],
        "layer_norm": ["LayerNorm", False, False],
        "batch_norm_l1": ["BatchNorm", True, False],
        "group_norm_l1": ["GroupNorm", True, False],
        "layer_norm_l1": ["LayerNorm", True, False],
        "batch_norm_l1_l2": ["BatchNorm", True, True]
    }

    device = torch.device("cuda" if cuda else "cpu")

    # Values for plotting graphs
    final_train_loss_values = []
    final_test_loss_values = []
    final_train_accuracy_values = []
    final_test_accuracy_values = []

    for key, value in models.items():
        print(f"=============== Running {key} Model ===============")
        # Run Epochs
        train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values = run_epochs(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            model=MODELS[value[0]], # Based on the value model instance will be selected from MODELS dictionary
            is_l1_norm=value[1],
            is_l2_norm=value[2]
        )

        wrong_predictions(test_loader=test_loader, model=MODELS[value[0]], device=device)  # Plot wrong predicted images

        final_train_loss_values.append(train_loss_values)
        final_test_loss_values.append(test_loss_values)
        final_train_accuracy_values.append(train_accuracy_values)
        final_test_accuracy_values.append(test_accuracy_values)
        print(f"=============== Finished {key} Model ===============")

    plot_accuracy_curves(final_train_loss_values, final_test_loss_values,
                         final_train_accuracy_values, final_test_accuracy_values)  # Values for plotting graphs


if __name__ == "__main__":
    run()
