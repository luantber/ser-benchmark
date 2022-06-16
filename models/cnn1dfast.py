from os import pread
from tkinter import Y
import torch
import torchmetrics
from torch.nn import functional as F
from torch import nn
from mef import Model


class CNN1DFast(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_input=1
        n_output=8
        stride=4
        n_channel=32

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

        self.accuracy = torchmetrics.Accuracy()
        self.accuracy_train = torchmetrics.Accuracy()

    def forward(self, step):
        #(128,100)   

        batch_size = step.shape[0]     
        step = step.view( batch_size,1,-1)

        x = self.conv1(step)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x,0.15)

        x = self.pool1(x)
        x = self.conv2(x)
        
        x = F.relu(x)
        x = F.dropout(x,0.15)

        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.dropout(x,0.15)

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.dropout(x,0.15)
        
        x = self.pool4(x)
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.view(batch_size,-1)
        x = self.fc1(x)
        return x

    def training_step(self, batch, idx):
        x, lengths, y = batch
        steps = []
        for t in range(len(x)):
            x_t = x[t]
            steps += [self(x_t)]

        steps_tensor = torch.stack(steps)

        # 6,128,8
        correct_list = []
        for i in range( len(lengths) ):
            correct = steps_tensor[ : lengths[i] , i , : ]
            correct_tensor_avg = correct.mean(dim=0)
            correct_list += [correct_tensor_avg]
        
        pred_tensor = torch.stack(correct_list)


        # print( pred_tensor.shape, y.shape )

        self.accuracy_train(pred_tensor, y)

        loss = F.cross_entropy(pred_tensor, y)
        self.log("loss", loss, batch_size=x.shape[1])
        self.log("acc", self.accuracy_train, batch_size=x.shape[1])

        return loss

    def validation_step(self, batch, idx):
        x, lengths, y = batch
        steps = []
        for t in range(len(x)):
            x_t = x[t]
            steps += [self(x_t)]

        steps_tensor = torch.stack(steps)

        # 6,128,8
        correct_list = []
        for i in range( len(lengths) ):
            correct = steps_tensor[ : lengths[i] , i , : ]
            correct_tensor_avg = correct.mean(dim=0)
            correct_list += [correct_tensor_avg]
        
        pred_tensor = torch.stack(correct_list)

        # breakpoint()
        
        # print(len(batch))
        # print(preds)
        # print(pred_tensor.shape, y.shape, "<<<")

        self.accuracy(pred_tensor, y)
        loss = F.cross_entropy(pred_tensor, y)
        self.log("val_loss", loss, batch_size=x.shape[1])
        self.log("val_acc", self.accuracy, batch_size=x.shape[1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
