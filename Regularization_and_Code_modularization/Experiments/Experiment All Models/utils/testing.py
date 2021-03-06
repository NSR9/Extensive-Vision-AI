from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Testing:
    def test1(model, device, test_loader, epoch):
        actuals = []
        predictions = []
        wrong_images = []
        wrong_label = []
        correct_label = []
        test_loss_values = []
        test_accuracy_values = []
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                if epoch == 20:
                    actuals.extend(target.view_as(pred))
                    predictions.extend(pred)
                    wrong_pred = pred.eq(target.view_as(pred)) == False
                    wrong_images.append(data[wrong_pred])
                    wrong_label.append(pred[wrong_pred])
                    correct_label.append(target.view_as(pred)[wrong_pred])

        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
        test_acc = 100.00 * correct / len(test_loader.dataset)
        test_accuracy_values.append(test_acc)
        test_loss_values.append(test_loss)
        return test_accuracy_values, test_loss_values, wrong_images, wrong_label, correct_label
