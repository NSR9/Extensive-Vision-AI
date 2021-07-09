import copy

from tqdm import tqdm 
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR,OneCycleLR

def cyclic_lr():
  max_lr = 0.02
  min_lr = 0.001
  epoch = 10
  criterion = nn.CrossEntropyLoss()
  model = resnet().to(device)
  momemtum = 0.9
  weight_decay=0.05,
  Lrtest_train_acc = []
  LRtest_Lr = []


  step = (max_lr - min_lr )/epoch
  lr = min_lr
  for e in range(epoch):
    testmodel = copy.deepcopy(model)
    optimizer = optim.SGD(testmodel.parameters(), lr=lr ,momentum=0.9,weight_decay=0.05 ) 
    lr += (max_lr - min_lr)/epoch
    testmodel.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data["image"].to(device), target.to(device)
        optimizer.zero_grad()
        y_pred =testmodel(data)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'epoch = {e+1} Lr = {optimizer.param_groups[0]["lr"]}  Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    Lrtest_train_acc.append(100*correct/processed)
    LRtest_Lr.append(optimizer.param_groups[0]['lr'])
  plt.plot(LRtest_Lr, Lrtest_train_acc)
  plt.ylabel('train Accuracy')
  plt.xlabel("Learning rate")
  plt.title("Lr v/s accuracy")
  plt.show()  