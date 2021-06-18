from tqdm import tqdm
import torch.nn.functional as F


def train(model, device, train_loader, optimizer, epoch, is_l1_norm, is_l2_norm, scheduler):
    model.train()
    train_loss = 0
    correct = 0
    processed = 0
    lambda_l1 = 0.001
    lambda_l2 = 0.001
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        l1 = 0
        l2 = 0

        # L1 Norm
        if is_l1_norm:
            for p in model.parameters():
              l1 = l1 + p.abs().sum()

        # L2 Norm
        if is_l2_norm:
            for p in model.parameters():
              l2 = l2 + p.pow(2.0).sum()

        loss = loss + lambda_l1 * l1 + lambda_l2 * l2
        loss.backward()
        optimizer.step()
        scheduler.step()
        processed += len(data)
        pbar.set_description(desc= f'Epoch{epoch} : Loss={loss.item()}  Accuracy={100*correct/processed:0.2f} Batch_id={batch_idx}')
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100.00*correct/len(train_loader.dataset)
    return train_acc, train_loss