from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
from data import Dataset
torch.manual_seed(0)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512,2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)

train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=100)

n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(validation_dataset)
N_train=len(train_dataset)
start_time = time.time()

Loss=0
start_time = time.time()
for epoch in range(n_epochs):
    for x, y in train_loader:

        model.train() 
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.data)
    correct=0
    for x_test, y_test in validation_loader:
        model.eval()
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
        
   
    accuracy=correct/N_test
    accuracy_list.append(accuracy)


plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()
