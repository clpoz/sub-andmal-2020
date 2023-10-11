import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2 as cv
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import models
import models2
from dataset import AndMalDataset
from dataset import AndMalDataset2
from matplotlib import pyplot as plt


ds = ["Adware","Backdoor","Banker","Dropper","FileInfector","PUA",
      "Ransomware","Riskware","Scareware","SMS","Spy","Trojan","Ben3"]
def pause():
    input('pausing..')

def train(epoch):
    total_loss = 0.0
    cnt = 0
    for i,(x,y) in enumerate(trainloader):
        x = x.cuda()
        #y = y.long().cuda()
        y = y.cuda()
        pred = model(x)
        # print(type(y),type(pred))
        # pause()
        optimizer.zero_grad()
        loss = criterion(pred,y).cuda()

        loss.backward()
        optimizer.step()
        cnt+=1
        total_loss+=loss.item()
    return total_loss


def mtest(epoch,show=False):
    t_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i,(x,y) in enumerate(testloader):
            x = x.cuda()
            y = y.long().cuda()
            pred = model(x)
            loss = criterion(pred,y).cuda()
            _,predicted = torch.max(pred.data,dim=1)
            total+=y.size(0)
            correct+=(predicted==y).sum().item()
            t_loss+=loss.item()
    acc = correct / total
    print('Accuracy on test set in %d: %.2f %%' % (epoch, 100 * acc))
    return t_loss,acc


def mtest2(epoch,show=False):
    t_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(testloader):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = criterion(pred,y).cuda()
            total+=y.size(0)
            t_loss+=loss.item()
            y = y.tolist()
            pred = pred.tolist()
            for j in range(len(y)):
                if (pred[j][0]<0.5 and y[j][0]==0.0) or (pred[j][0]>=0.5 and y[j][0]==1.0):
                    correct+=1
    acc =  correct / total
    print('Accuracy on test set in %d: %.2f %%' % (epoch, 100*acc))
    return t_loss,acc


def make_matrix(model,num_class,bs):
    matrix = np.zeros((num_class,num_class),dtype=np.int)
    TP = np.zeros(num_class,dtype=np.int)
    FP = np.zeros(num_class, dtype=np.int)
    TN = np.zeros(num_class, dtype=np.int)
    FN = np.zeros(num_class, dtype=np.int)
    precision = np.zeros(num_class)
    recall = np.zeros(num_class)
    acc = None
    tot=0

    with torch.no_grad():
        for i,(x,y) in enumerate(testloader):
            x = x.cuda()
            pred = model(x)
            _, predicted = torch.max(pred.data, dim=1)
            y =y.tolist()
            predicted = predicted.tolist()
            for j in range(len(y)):
                matrix[y[j]][predicted[j]]+=1
            tot+=len(y)

    print(matrix)
    for i in range(num_class):
        TP[i] = matrix[i][i]
        FP[i] = sum([matrix[j][i] for j in range(num_class)])-matrix[i][i]
        TN[i] = tot - sum(matrix[i]) - sum([matrix[j][i] for j in range(num_class)])\
                +matrix[i][i]
        FN[i] = sum(matrix[i])-matrix[i][i]

        precision[i] = TP[i] / (TP[i]+FP[i])
        recall[i] = TP[i] / (TP[i]+FN[i])

    acc = sum(TP) / tot
    print('Accuracy: %.2f'%acc)
    print('   precision   recall')
    for i in range(num_class):
        print('%s : %.2f %.2f'%(ds[i],precision[i],recall[i]))


def make_matrix2(model,bs):
    matrix = np.zeros((2,2),dtype=np.int)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    tot=0

    score = []
    label = []
    with torch.no_grad():
        for i,(x,y) in enumerate(testloader):
            x = x.cuda()
            predicted = model(x)
            l = len(y)
            y =y.view(l).tolist()
            predicted = predicted.view(l).tolist()
            label.extend(y)
            score.extend(predicted)
            predicted = [1.0 if pred>0.5 else 0.0 for pred in predicted]
            for j in range(l):
                if y[j]==1.0:
                    if predicted[j]==1.0:
                        TP+=1
                    else:
                        FN+=1
                else:
                    if predicted[j]==1.0:
                        FP+=1
                    else:
                        TN+=1
            tot+=l
    acc = (TP+TN)/(TP+FP+FN+TN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print('TP FP\nFN TN')
    print(TP,FP)
    print(FN,TN)
    print('Accuracy: %.2f'%acc)
    print('precision %.2f  recall %.2f'%(precision,recall))
    make_roc(label=label,score=score)


def make_roc(label,score):
    XY = []

    for thr in range(10000):
        threshold = thr / 10000
        FPR,TPR = countXY(label,score,threshold)
        XY.append([FPR,TPR])
    XY.sort(key=lambda a:a[0])
    XY[-1] = [1.0,1.0]

    X = [xy[0] for xy in XY]
    Y = [xy[1] for xy in XY]

    auc = 0
    for i in range(1,10000):
        auc+=(X[i]-X[i-1])*(Y[i]+Y[i-1])*0.5
    print('auc',auc)
    plt.plot(X,Y)
    plt.title('AUC= %.4f'%auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.show()

def countXY(label,score,threshold):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    score = [1.0 if pred >= threshold else 0.0 for pred in score]
    for j in range(len(label)):
        if label[j] == 1.0:
            if score[j] == 1.0:
                TP += 1
            else:
                FN += 1
        else:
            if score[j] == 1.0:
                FP += 1
            else:
                TN += 1

    FPR = 0.0 if FP==0 else FP/(FP+TN)
    TPR = 0.0 if TP==0 else TP/(TP+FN)
    # if threshold==0.0:
    #     print(TP, FP)
    #     print(FN, TN)
    #     print(FPR,TPR)
    return FPR,TPR

if __name__=='__main__':
    trainset = AndMalDataset2('data/32/andmal_train.h5',is_Train=True)
    testset = AndMalDataset2('data/32/andmal_test.h5', is_Train=False)
    trainloader = DataLoader(dataset=trainset,shuffle=True,batch_size=32,num_workers=0)
    testloader = DataLoader(dataset=testset,shuffle=True,batch_size=32,num_workers=0)

    min_loss = 50
    max_acc = 0.90
    model = models2.Model2(in_features=9503,out_features=1,in_channels=1,out_channels=128)
    model=model.cuda()
    #model = torch.load('model2.pth')

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    #criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.5)

    X = range(1,51)
    Y = []
    for epoch in range(50):
        loss = train(epoch)
        Y.append(loss)
        t_loss, acc= mtest2(epoch)
        if acc > max_acc:
            min_loss = t_loss
            max_acc=acc
            torch.save(model,model.modelName+'.pth')

        print('training epoch :%d  loss: %.4f  test_loss: %.4f' % (epoch, loss, t_loss))

    t_loss,acc= mtest2(0)
    print('test loss:%.4f '%t_loss)
    plt.plot(X,Y)
    plt.xlabel('training epoch')
    plt.ylabel('training loss')
    plt.show()
    #make_matrix(model=model,num_class=12,bs=32)
    #make_matrix2(model=model,bs=32)
