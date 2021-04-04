import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

def accuracy_(preds, labels):
    return torch.sum(preds == labels).item() / len(preds)

batch_size = 100

model = MnistModel(784, 10)

dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)

train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)

val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size, sampler=val_sampler)

for images, labels in train_loader:
    outputs = model(images)
    break

print('outputs.shape: ', outputs.shape)
print('Sample outputs: ', outputs[:2].data)

probs = F.softmax(outputs, dim=1)
print('Sample probabilities: ', probs[:2].data)
print('Sum: ', torch.sum(probs[0]).item())

max_probs, preds = torch.max(probs, dim=1)
print(max_probs)
print(preds)
print(labels)

print(accuracy_(preds, labels))

loss_func = F.cross_entropy

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

 
def loss_batch(model, loss_func, img, label, opt=None, metric=None):
    preds = model(img)
    loss = loss_func(preds, label)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, label)

    return loss.item(), len(img), metric_result

def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
                   for xb, yb in valid_dl]
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, metric=None):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            loss,_,_ = loss_batch(model, loss_func, xb, yb, opt)

        result = evaluate(model, loss_func, valid_dl, metric)
        val_loss, total, val_metric = result

        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, val_loss))
        else:
            print('Epoch[{}/{}], Loss: {:.4f}, {}: {:.4f}'
                  .format(epoch+1, epochs, val_loss, metric.__name__, val_metric))

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

fit(5, model, loss_func, optimizer, train_loader, val_loader, accuracy)
print('\n\n')

for i in range(10):
    img, label = test_dataset[i]
    print('Label: ', label,', Predicated: ', predict_image(img, model))
    
