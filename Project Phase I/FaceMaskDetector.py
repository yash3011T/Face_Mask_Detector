#!/usr/bin/env python
# coding: utf-8

# In[132]:


#Importing libraries
import os
import cv2
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch import Tensor

import warnings 
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt


# In[133]:


# Path of training and testing dataset
DIR_TRAIN = "C:\\Users\\yasht\\OneDrive\\Desktop\\Study\\AI\\Assignments\\AI Project Phase I\\TRAIN"




# In[134]:


# Labels (classes) to differentiate the images in these categories
label_dict = {
    0: "Person with Cloth Mask", 
    1: "Person with N-95 Mask", 
    2: "Person with Surgical Mask",
    3: "Person without Mask",
    4: "Person with incorrect Mask"
}

# Labels to display on the confussion matrix
labels_list = ["Person with Cloth Mask","Person with N-95 Mask","Person with Surgical Mask","Person without Mask","Person with incorrect Mask"]


# In[155]:


#Displaying total images in each class and total number of images overall
classes = os.listdir(DIR_TRAIN)
print("Total Classes: ",len(classes),"\n")
#Counting total images in each class

total = 0
individual_count = []
for _class in classes:
    individual_count.append(len(os.listdir(DIR_TRAIN +"/"+_class)))
    total += len(os.listdir(DIR_TRAIN + "/"+_class))

for i in range(0,len(individual_count)):
  print(classes[i],": ", individual_count[i])
print("\n")
print("Total : ", total, "\n")


# In[156]:


train_imgs = []
test_imgs = []

for _class in classes:
    
    for img in os.listdir(DIR_TRAIN +"/"+ _class):
        train_imgs.append(DIR_TRAIN + _class + "/" + img)
        

class_to_int = {classes[i] : i for i in range(len(classes))}


# In[157]:


#Loading Classification Dataset


transform = T.Compose([T.Resize((224,224)),
                                T.ToTensor()])

#training_data = ImageFolder(root = DIR_TRAIN, transform = transform)
#testing_data = ImageFolder(root = DIR_TEST, transform = transform)

train_dataset = ImageFolder(root = DIR_TRAIN, transform = transform)
dataset = len(train_dataset)
training_data, testing_data = torch.utils.data.random_split(train_dataset, [dataset-int(0.25*dataset) , int(0.25*dataset)])



#Data Loader
train_random_sampler = RandomSampler(training_data)
test_random_sampler = RandomSampler(testing_data)


train_data_loader = DataLoader(
    dataset = training_data,
    batch_size = 16,
    num_workers = 4,
    shuffle= True
)


test_data_loader = DataLoader(
    dataset = testing_data,
    batch_size = 16,
    num_workers = 4,
    shuffle=True
)

print(len(training_data))
print(len(testing_data))


# In[169]:


class MaskDetect(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # convolution layer 1
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # convolution layer 2
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            
            # convolution layer 3
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # convolution layer 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # convolution layer 5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(12544, 7)
        )

    # forward pass to readjust weights
    def forward(self, x):
        x = self.cnn_layers(x)
        print(x.shape())
        x = x.view(x.size(0), -1)
        #         print(x.size())
        x = self.linear_layers(x)
        return x


# In[159]:


#Training Details
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.75)
criterion = nn.CrossEntropyLoss()

train_loss = []
train_accuracy = []


epochs = 20


# In[160]:


#Defining function of accuracy calculation
def calc_accuracy(true,pred):
    pred = F.softmax(pred, dim = 1)
    true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
    acc = (true.argmax(-1) == pred.argmax(-1)).float().detach().numpy()
    acc = float((100 * acc.sum()) / len(acc))
    return round(acc, 4)


# In[165]:


#Training the model

testing_accuracy = []

for epoch in range(epochs):
    model.train()    
    start = time.time()
    
    #Epoch Loss & Accuracy
    train_epoch_loss = []
    train_epoch_accuracy = []
    _iter = 1
    
    #Training
    for images, labels in train_data_loader:
        
        images = images.to(device)
        labels = labels.to(device)
        
        #Reset Grads
        optimizer.zero_grad()
  
        #Forward ->
        preds = model(images)
        
        #Calculate Accuracy
        acc = calc_accuracy(labels.cpu(), preds.cpu())
        
        #Calculate Loss & Backward, Update Weights (Step)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        #Append loss & accuracy
        loss_value = loss.item()
        train_epoch_loss.append(loss_value)
        train_epoch_accuracy.append(acc)
        
        if _iter % 100 == 0:
            print("> Iteration {} < ".format(_iter))
            print("Iter Loss = {}".format(round(loss_value, 4)))
            print("Iter Accuracy = {} % \n".format(acc))
        
        _iter += 1

    end = time.time()
    
    train_epoch_loss = np.mean(train_epoch_loss)
    train_epoch_accuracy = np.mean(train_epoch_accuracy)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    
    #Print Epoch Statistics
    print("** Epoch {} ** - Epoch Time {}".format(epoch+1, int(end-start)))
    print("Train Loss = {}".format(round(train_epoch_loss, 4)))


# In[170]:


torch.save(model.state_dict(), "C:\\Users\\yasht\\OneDrive\\Desktop\\Study\\AI\\Assignments\\AI Project Phase I\\Trained_Model.pt")


# In[171]:


plt.plot(train_loss, label='Training loss')
plt.title('Loss at the end of each epoch')
plt.legend()
plt.show()


# In[172]:


#Testing the model
testing_accuracy = []
predictions_list = []
accurate_list = []

with torch.no_grad():

  for images, labels in test_data_loader:      
        model.eval()
        images = images.to(device)
        labels = labels.to(device)
        _, pred_values = torch.max(model(images), dim=1)
        predictions_list.extend(pred_values.detach().cpu().numpy())
        accurate_list.extend(labels.detach().cpu().numpy())
        #Forward ->
        preds = model(images)
        #Calculate Accuracy
        acc = calc_accuracy(labels.cpu(), preds.cpu())
        testing_accuracy.append(acc);

print("Final Accuracy: ", np.mean(testing_accuracy),"\n")
print("Testing Classification Report")
print(classification_report(accurate_list, predictions_list),"\n")
print("Confusion Matrix:")
# plt.figure()
confusion_matrix_instance = confusion_matrix(accurate_list, predictions_list)
plt.imshow(confusion_matrix_instance, interpolation='nearest', cmap=plt.cm.Pastel2)
for (x_cordinate, y_cordinate), val in np.ndenumerate(confusion_matrix_instance):
    plt.text(x_cordinate, y_cordinate, val, ha='center', va='center')
plt.title('Testing Confusion matrix')
plt.ylabel('Actual labels')
plt.xlabel('Predicted labels')
randomized_val = np.arange(len(labels_list))
plt.xticks(randomized_val, labels_list, rotation=60)
plt.yticks(randomized_val, labels_list)
plt.show()





