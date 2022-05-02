from tkinter import Y
import torch
import torchmetrics
from torch.nn import functional as F
from torch import nn
from mef import Model


class CNN1D(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.c1 = torch.nn.Conv1d(1, 120, 80, 10)
        self.c2 = torch.nn.Conv1d(120, 40, 10)
        self.c3 = torch.nn.Conv1d(40, 64, 10)
        self.l1 = torch.nn.Linear(64 * 42, 8)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, series, length):

        outs = []
        # print(series.shape, length)
        for i in range(length):
            wave = series[i]

            out = torch.relu(self.c1(wave.view(1, -1)))
            out = torch.max_pool1d(out, 4)
            out = torch.relu(self.c2(out))
            out = torch.max_pool1d(out, 2)
            out = torch.relu(self.c3(out))
            out = torch.max_pool1d(out, 2)
            # print(out.shape)
            out = torch.relu(self.l1(out.view(-1)))

            outs += [out]

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
        loss = F.cross_entropy(pred_tensor, y)
        self.log("loss", loss)

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
