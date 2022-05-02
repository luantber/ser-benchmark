from tkinter import Y
import torch
import torchmetrics
from torch.nn import functional as F
from torch import nn
from mef import Model


class CNN1D(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_input=1
        n_output=8
        stride=16
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

    def forward(self, series, length):

        outs = []
        # print(series.shape, length)
        for i in range(length):
            wave = series[i]
            
            out = wave.view(1,1, -1)
            
            x = self.conv1(out)
            x = F.relu(x)
            x = self.pool1(x)

            x = F.dropout(x,0.15)

            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)

            x = F.dropout(x,0.15)

            x = self.conv3(x)
            x = F.relu(x)
            x = self.pool3(x)

            x = F.dropout(x,0.15)

            x = self.conv4(x)
            x = F.relu(x)
            x = self.pool4(x)

            x = F.dropout(x,0.15)
            
            x = F.avg_pool1d(x, x.shape[-1])
            x = x.flatten()
            x = self.fc1(x)
            
            outs += [x]

        out_tensor = torch.stack(outs)
        out_mean = torch.mean(out_tensor, dim=0)
        # print(out_mean, "mean")
        return out_mean

    def training_step(self, batch, idx):
        # ( B , T , N)
        x, lengths, y = batch
        preds = []
        for i in range(len(x)):
            x_i, length_i = x[i], lengths[i]
            preds += [self(x_i, length_i)]

        pred_tensor = torch.stack(preds)

        # print( pred_tensor.shape, y.shape )

        self.accuracy_train(pred_tensor, y)

        loss = F.cross_entropy(pred_tensor, y)
        self.log("loss", loss)
        self.log("acc", self.accuracy_train)

        return loss

    def validation_step(self, batch, idx):
        x, lengths, y = batch
        preds = []
        for i in range(len(x)):
            x_i, length_i = x[i], lengths[i]
            preds += [self(x_i, length_i)]

        pred_tensor = torch.stack(preds)
        
        # print(len(batch))
        # print(preds)
        # print(pred_tensor.shape, y.shape, "<<<")

        self.accuracy(pred_tensor, y)
        loss = F.cross_entropy(pred_tensor, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
