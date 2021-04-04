import torch
import numpy as np
import math
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

inputs = np.array(
    [[73,67,43],
     [91,88,64],
     [87,134,58],
     [102,43,37],
     [69,96,70],
     [73,67,43],
     [91,88,64],
     [87,134,58],
     [102,43,37],
     [69,96,70],
     [73,67,43],
     [91,88,64],
     [87,134,58],
     [102,43,37],
     [69,96,70]],dtype='float32')

targets = np.array(
    [[56,70],
     [81,101],
     [119,133],
     [22,37],
     [103,119],
     [56,70],
     [81,101],
     [119,133],
     [22,37],
     [103,119],
     [56,70],
     [81,101],
     [119,133],
     [22,37],
     [103,119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

train_ds = TensorDataset(inputs,targets)

batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = nn.Linear(3, 2)

loss_fn = F.mse_loss

opt = torch.optim.SGD(model.parameters(), lr=1e-5)

def fit(num_epochs, model, loss_fn, opt):

    for epoch in range(num_epochs):

        for xb,yb in train_dl:
            
            preds = model(xb)

            loss = loss_fn(preds, yb)

            loss.backward()

            opt.step()

            opt.zero_grad()
            
        if(epoch+1)%10 == 0:
            print('Epoch[{}/{}], Loss:{:.4f}'
                  .format(epoch+1, num_epochs, loss.item()))
    
        
fit(1000, model, loss_fn,opt)



























