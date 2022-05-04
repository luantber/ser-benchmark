from pyparsing import col
from yaml import load
from models.cnn1dfast import CNN1DFast
from datasets.ravdess import Ravdess
import torch 
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from datasets.utils import collate_padded_no
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler

seed = 42
# model = CNN1DFast("cnn1d_test")

dataset = Ravdess("datasets/ravdess/train/train.csv", "datasets/ravdess/train/audios")


loader = DataLoader(dataset,batch_size=5,collate_fn=collate_padded_no)

for i in range(600):
    # print(i)
    for batch in loader:
        a = batch[0].shape



        # print( batch[0].shape )

        # break

# 1m35,914s
# 36s 0wks