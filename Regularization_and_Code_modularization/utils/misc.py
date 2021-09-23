"""

"""
import torch.cuda
import train_utility, test_utiliy


def get_model_summary(model):
    """
    Summary of model will be printed
    Args:
        model: Model

    Returns: summary of model

    """
    from torchsummary import summary
    model_summary = summary(model, input_size=(1, 28, 28))
    return model_summary


def load_optimizer(model):
    """
    define optimizer and scheduler
    Args:
        model: model

    Returns: optimizer and scheduler

    """
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.7)

    return optimizer


def get_cuda():
    seed = 1
    cuda = torch.cuda.is_available()
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
    return cuda


def run_epochs(train_loader, test_loader, device, model, is_l1_norm, is_l2_norm):
    from torch.optim.lr_scheduler import StepLR, OneCycleLR

    model = model.to(device) # Get Model Instance
    summary = get_model_summary(model=model)
    print(summary)

    optimizer = load_optimizer(model)
    scheduler = OneCycleLR(optimizer, max_lr=0.015, epochs=20, steps_per_epoch=len(train_loader))

    train_loss_values = []
    test_loss_values = []
    train_accuracy_values = []
    test_accuracy_values = []

    for epoch in range(1, 21):
        train_acc, train_loss = train_utility.train(
            model=model,
            device=device,
            train_loader=train_loader,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            is_l1_norm=is_l1_norm,
            is_l2_norm=is_l2_norm
        )

        test_acc, test_loss = test_utiliy.test(
            model=model,
            device=device,
            test_loader=test_loader,
            epoch=epoch
        )

        train_accuracy_values.append(train_acc)
        train_loss_values.append(train_loss)

        test_accuracy_values.append(test_acc)
        test_loss_values.append(test_loss)

    return train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values




