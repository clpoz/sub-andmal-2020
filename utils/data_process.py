import pandas as pd
import numpy as np
import os
import csv
import h5py
import torch.nn as nn
import torch
import json
import h5py

with open('datasets4.json','r') as f:
    dataconf = json.load(f)
datadir = "${FILE_PATH}/Malicious-CSVs"

ds = ["Adware","Backdoor","Banker","Dropper","FileInfector","PUA",
      "Ransomware","Riskware","Scareware","SMS","Spy","Trojan","Ben3"]

def make_h5():
    ps = ["Ben3"]
    trainX = []
    testX= []
    trainY = []
    testY = []

    label = 0
    "len 9503"
    for d in ds:
        filepath = datadir+"/"+d+".csv"
        print(d,label)
        data = np.genfromtxt(filepath,delimiter=',',dtype=str,max_rows=dataconf[d][0]+dataconf[d][1])
        #data = np.loadtxt(f,str,delimiter=',',)
        data = data[:,1:]
        data = data.astype(np.int32)
        trainX.extend(data[:dataconf[d][0],:])
        testX.extend(data[dataconf[d][0]:dataconf[d][0]+dataconf[d][1],:])
        trainY.extend([label for i in range(dataconf[d][0])])
        testY.extend([label for i in range(dataconf[d][1])])
        label+=1

    trainX = np.array(trainX,dtype=np.int32)
    trainY = np.array(trainY,dtype=np.int32)
    testX = np.array(testX,dtype=np.int32)
    testY = np.array(testY,dtype=np.int32)
    print(trainX.shape,trainY.shape,testX.shape,testY.shape)

    save_path = "../data/32/andmal_train.h5"
    with h5py.File(save_path,"w") as f:
        f['trainX']=trainX
        f['trainY']=trainY
        # f['testX']=testX
        # f['testY']=testY
    save_path = "../data/32/andmal_test.h5"
    with h5py.File(save_path,"w") as f:
        f['testX']=testX
        f['testY']=testY

if __name__=='__main__':
    #make_h5()
    f = h5py.File('../data/32/andmal_train.h5','r')
    trainX = np.array(f['trainX'],dtype=np.float32)
    trainY = f['trainY']
    f = h5py.File('../data/32/andmal_test.h5','r')
    testX = np.array(f['testX'],dtype=np.float32)
    testY = f['testY']
    print(trainX.shape,trainY.shape,testX.shape,testY.shape)
    print(type(trainX))
