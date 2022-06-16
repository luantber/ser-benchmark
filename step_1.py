from datasets.ravdess import Ravdess
from datasets.utils import collate_padded_no

# from models.cnn1d import CNN1D
from models.cnn1dfast import CNN1DFast
from models.m11 import M11


import mef

dataset = Ravdess("datasets/ravdess/train/train.csv", "datasets/ravdess/train/audios")

settings = {
    "cnn1d_shorts": mef.Setting(
        CNN1DFast,
        batch_size=256,
        epochs=800,
        dataloader_args={"collate_fn": collate_padded_no},
    ),

    "m11": mef.Setting(
        M11,
        batch_size=256,
        epochs=500,
        dataloader_args={"collate_fn": collate_padded_no},
    ),
   
}

exp = mef.Experiment(settings=settings, dataset=dataset)
# results = exp.test("m11", 42)
# print(results)
results = exp.test("cnn1d_shorts", 42)
print(results)
