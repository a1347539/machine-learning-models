import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class MnistModel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        half = math.floor(in_size/2)
        quater = math.floor(half/2)
        self.linear1 = nn.Linear(in_size, half)
        self.linear2 = nn.Linear(half, quater)
        self.linear3 = nn.Linear(quater, out_size)

    def forward(self, xb):
        #flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        #get intermediate outputs using hidden layer
        out = self.linear1(xb)
        #apply activation function
        out = F.relu(out)
        #get prediction using out layer
        out = self.linear2(out)

        out = F.relu(out)

        out = self.linear3(out)
        return out

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)    

dataset = MNIST(root='data/', download=True, transform=ToTensor())

test_dataset = MNIST(root='data/', train=False, transform=ToTensor())

def split_indices(n, val_pct):
    n_val = int(n * val_pct)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

train_indices, val_indices = split_indices(len(dataset), 0.2)

batch_size = 100

train_sampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset, batch_size, sampler=train_sampler)

valid_sampler = SubsetRandomSampler(val_indices)
valid_dl = DataLoader(dataset, batch_size, sampler=valid_sampler)

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        loss.backward()

        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)
        
    return loss.item(), len(xb), metric_result

def evaluate(model, loss_fn, valid_dl, metric):
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
        


def fit(epochs, lr, model, loss_fn, train_dl, valid_dl, metric=None, opt_fn=None):
    losses, metrics = [], []
    if opt_fn is None: opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in train_dl:
            loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)

        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        losses.append(val_loss)
        metrics.append(val_metric)

        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, val_loss))
        else:
            print('Epoch[{}/{}], Loss: {:.4f}, {}: {:.4f}'
                  .format(epoch+1, epochs, val_loss, metric.__name__, val_metric))
    return losses, metrics
        
input_size = 784
num_classes = 10
model = MnistModel(input_size, out_size=num_classes)
to_device(model, device)

'''
fit(20, 0.1, model, F.cross_entropy,
                       train_dl, valid_dl, metric=accuracy)
'''

model.load_state_dict(torch.load('model2'))

def prediction(image):
    output = model(image)
    prob, pred = torch.max(output, dim=1)
    return pred.item()
'''
hit=0
for i in range(500,1000):
    label, pred = prediction(test_dataset[i])
    
    if label == pred:
        hit+=1
        
    print('label: {}, prediction: {}'.format(label, pred))

print('accuracy: {}/{}'.format(hit, i+1-500))
'''
image, label = test_dataset[650]
#plt.imshow(image[0], cmap="gray")
img = Image.open('1.png').convert('L')

img = ToTensor()(img).unsqueeze(0)

img = img[0]
plt.imshow(img[0], cmap='gray')

pred = prediction(img)

print('prediction: {}'.format(pred))













plt.show()




