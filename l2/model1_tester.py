import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn as nn

test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

class MnistModel(nn.Module):
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

model = MnistModel(784, 10)
model.load_state_dict(torch.load('model1'))

hit = 0
for i in range(100):
    img, label = test_dataset[i]
    pred = predict_image(img, model)
    if label == pred:
        hit+=1
    print('Label: ', label,', Predicated: ', pred)
    print()
print(hit, i)
print(hit/i)
