from datasets.ravdess import Ravdess
from datasets.utils import collate_padded_no

# from models.cnn1d import CNN1D
from models.cnn1dfast import CNN1DFast

import mef

import pandas as pd

dataset = Ravdess("datasets/ravdess/train/train.csv", "datasets/ravdess/train/audios")

# dataframe = pd.DataFrame(dataset[0][0][1])
# print(dataframe.describe())

settings = {
    "cnn1d": mef.Setting(
        CNN1DFast,
        batch_size=128,
        epochs=1500,
        dataloader_args={"collate_fn": collate_padded_no},
    ),
}

exp = mef.Experiment(settings=settings, dataset=dataset)
results = exp.test("cnn1d", 42)

print(results)
