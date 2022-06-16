from datasets.ravdess import Ravdess
from datasets.utils import collate_padded_no

from models.cnn1dfast import CNN1DFast
from models.m11 import M11


import warnings

warnings.filterwarnings("ignore")

import mef


dataset = Ravdess("datasets/ravdess/train/train.csv", "datasets/ravdess/train/audios")

settings = {
    "cnn1d_shorts_128": mef.Setting(
        CNN1DFast,
        batch_size=128,
        epochs=1000,
        dataloader_args={"collate_fn": collate_padded_no},
    ),

    "m11_128": mef.Setting(
        M11,
        batch_size=128,
        epochs=500,
        dataloader_args={"collate_fn": collate_padded_no},
    ),

    "m11_256": mef.Setting(
        M11,
        batch_size=256,
        epochs=600,
        dataloader_args={"collate_fn": collate_padded_no},
    ),
   
}

exp = mef.Experiment(settings=settings, dataset=dataset)

# 0,1,2,3,4,5,6,7,8,9
# exp.run_model("cnn1d_shorts_128", iterations_range=range(20), kfold=6)
# exp.run_model("m11_128", iterations_range=range(10), kfold=6)
exp.run_model("m11_256", iterations_range=range(10), kfold=6)
