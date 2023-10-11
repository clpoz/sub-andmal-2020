import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2 as cv
import os
import numpy as np
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F



class Model3(torch.nn.Module):

    def __init__(self,in_features,out_features,in_channels,out_channels):
        super(Model3, self).__init__()
        self.modelName = '2model3'
        self.maxpool = torch.nn.MaxPool1d(2)
        self.avgpool = torch.nn.AvgPool1d(2)
        self.sig = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv1d(1,16,kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv1d(16,64,kernel_size=5)
        self.conv3 = torch.nn.Conv1d(64,128,kernel_size=3,padding=1)
        self.conv4 = torch.nn.Conv1d(128,16,kernel_size=3,padding=1)
        self.bn1d1 = torch.nn.BatchNorm1d(in_channels)
        self.bn1d2 = torch.nn.BatchNorm1d(16)
        self.bn1d3 = torch.nn.BatchNorm1d(64)
        self.bn1d4 = torch.nn.BatchNorm1d(128)
        self.fc = torch.nn.Linear(9488,1)
        self.relu = torch.nn.ReLU()


    def forward(self,x):
        batch_size = x.size(0)
        #x = x.view(batch_size,1,-1)
        x = x.unsqueeze(1)
        x = self.bn1d1(x)
        x = self.relu(self.avgpool(self.conv1(x)))
        x = self.bn1d2(x)
        x = self.relu(self.avgpool(self.conv2(x)))
        x = self.bn1d3(x)
        x = self.relu(self.avgpool(self.conv3(x)))
        x = self.bn1d4(x)
        x = self.relu(self.avgpool(self.conv4(x)))
        # torch.Size([32, 16, 593])
        x = x.view(batch_size,-1)
        x = self.sig(self.fc(x))

        return x
