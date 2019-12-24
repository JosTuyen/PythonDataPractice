import sklearn.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h0 = nn.Linear(4,10)
        self.out = nn.Linear(10,3)

    def forward(self, x):
        x = self.h0(x)
        x = F.tanh(x)
        x = self.out(x)
        x = F.tanh(x)
        return x

def calAccuracy(targets, predict):
    return (targets.argmax(1) == predict).sum().item() / predict.size()[0] * 100

def main():
    data_set = sklearn.datasets.load_iris()

    inputs = np.array(data_set["data"],dtype=np.float32)

    targets = np.zeros((inputs.shape[0],3))
    targets[data_set["target"] == 0, 0] = 1
    targets[data_set["target"] == 0, 1] = -1
    targets[data_set["target"] == 0, 2] = -1
    targets[data_set["target"] == 1, 0] = -1
    targets[data_set["target"] == 1, 1] = 1
    targets[data_set["target"] == 1, 2] = -1
    targets[data_set["target"] == 2, 0] = -1
    targets[data_set["target"] == 2, 1] = -1
    targets[data_set["target"] == 2, 2] = 1

    train_inputs,test_inputs,train_targets,test_targets = train_test_split(inputs, targets, test_size=0.3)

    train_inputs = torch.Tensor(train_inputs)
    test_inputs = torch.Tensor(test_inputs)
    train_targets = torch.Tensor(train_targets)
    test_targets = torch.Tensor(test_targets)

    lr_list = [0.02*x for x in range(1,20)]
    momentum_list = [0.1*x for x in range(1,10)]

    results = []
    for lr_val in lr_list:
        for momen in momentum_list:
            ANN = Net()
            optimizer = torch.optim.SGD(ANN.parameters(), lr=lr_val, momentum=momen)
            temp_ms_error = []
            for epoch in range(100):
                optimizer.zero_grad()
                out = ANN(train_inputs)
                loss = F.mse_loss(out,train_targets)
                loss.backward()
                optimizer.step()
                temp_ms_error.append(loss.item())
            results.append([lr_val,momen,mean(temp_ms_error)])
    results.sort(key=lambda x: x[2])
    best_lr = results[0][0]
    best_momentum = results[0][1]
    train_mse = []
    train_acc = []
    test_mse = []
    test_acc = []
    ANN = Net()
    optimizer = torch.optim.SGD(ANN.parameters(), lr=best_lr, momentum=best_momentum)
    for epoch in range(500):
        optimizer.zero_grad()
        out = ANN(train_inputs)
        loss = F.mse_loss(out,train_targets)
        predict = out.argmax(1)
        acc = calAccuracy(train_targets, predict)
        train_mse.append(loss.item())
        train_acc.append(acc)
        
        test_out = ANN(test_inputs)
        test_loss = F.mse_loss(test_out,test_targets)
        test_predict = test_out.argmax(1)
        test_acc_value = calAccuracy(test_targets, test_predict)
        test_mse.append(test_loss.item())
        test_acc.append(test_acc_value)
        loss.backward()
        optimizer.step()
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.plot(range(500), test_acc, color='green', label="Test Accuracy")
    plt.plot(range(500), train_acc, color='red', label="Train Accuravy")
    plt.legend()
    plt.savefig('Train_Test_Accuracy.png')
    plt.show()
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.plot(range(500), test_mse, color='black', label="Test MSE")
    plt.plot(range(500), train_mse, color='blue', label="Train MSE")
    plt.legend()
    plt.savefig('Train_Test_MSE.png')
    plt.show()

if __name__ == '__main__':
    main()