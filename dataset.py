import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class AndMalDataset(Dataset):
    def __init__(self,data_path,is_Train=True):
        super(AndMalDataset, self).__init__()
        f = h5py.File(data_path,'r')
        if is_Train:
            self.X =  np.array(f['trainX'],dtype=np.float32)
            self.Y =  f['trainY']
        else:
            self.X =  np.array(f['testX'],dtype=np.float32)
            self.Y =  f['testY']

        self.X = torch.from_numpy(self.X)
        self.len = len(self.Y)
        print(self.len)
        print(self.X.shape,self.Y.shape)

    def __getitem__(self, item):
        return self.X[item],self.Y[item]

    def __len__(self):
        return self.len

class AndMalDataset2(Dataset):
    def __init__(self,data_path,is_Train=True):
        super(AndMalDataset2, self).__init__()
        f = h5py.File(data_path,'r')
        if is_Train:
            self.X = np.array(f['trainX'],dtype=np.float32)
            self.Y = f['trainY']
        else:
            self.X = np.array(f['testX'],dtype=np.float32)
            self.Y = f['testY']

        self.X = torch.from_numpy(self.X)
        self.Y = np.array(self.Y)
        self.len = len(self.Y)
        for i in range(self.len):
            self.Y[i]= 0 if self.Y[i]<12 else 1
        self.Y = torch.from_numpy(self.Y)
        self.Y = self.Y.view(self.len,1)
        print(self.len)
        print(self.X.shape,self.Y.shape)

    def __getitem__(self, item):
        return self.X[item],self.Y[item].float()

    def __len__(self):
        return self.len



if __name__=='__main__':
    trainset = AndMalDataset2('data/32/andmal_train.h5', is_Train=True)
    testset = AndMalDataset2('data/32/andmal_test.h5', is_Train=False)

