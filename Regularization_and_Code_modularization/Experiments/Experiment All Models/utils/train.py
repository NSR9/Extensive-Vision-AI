from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm


class Train:
    def no_reg(model, device, train_loader, optimizer, epoch):
        train_loss_values = []
        train_accuracy_values = []
        model.train()
        train_loss = 0
        correct = 0
        processed = 0
        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            processed += len(data)
            pbar.set_description(
                desc=f"Epoch{epoch} : Loss={loss.item()}  Accuracy={100*correct/processed:0.2f} Batch_id={batch_idx}"
            )
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_accuracy_values.append(100.00 * correct / len(train_loader.dataset))
        train_loss_values.append(train_loss / len(train_loader))
        return train_accuracy_values, train_loss_values

    def L1(model, device, train_loader, optimizer, epoch):
        train_loss_values = []
        train_accuracy_values = []
        model.train()
        train_loss = 0
        correct = 0
        processed = 0
        lambda_l1 = 0.001
        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            train_loss += loss.item()
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1 * l1
            loss.backward()
            optimizer.step()
            processed += len(data)
            pbar.set_description(
                desc=f"Epoch{epoch} : Loss={loss.item()}  Accuracy={100*correct/processed:0.2f} Batch_id={batch_idx}"
            )
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        train_accuracy_values.append(100.00 * correct / len(train_loader.dataset))
        train_loss_values.append(train_loss / len(train_loader))
        return train_accuracy_values, train_loss_values
