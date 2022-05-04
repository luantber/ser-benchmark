from models.cnn1dfast import CNN1DFast
from datasets.ravdess import Ravdess
import torch 
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from datasets.utils import collate_padded_no
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler

seed = 42
model = CNN1DFast("cnn1d_test")

dataset = Ravdess("datasets/ravdess/train/train.csv", "datasets/ravdess/train/audios")

size_train = int(len(dataset) * 0.75)
size_test = len(dataset) - size_train
args = {"collate_fn": collate_padded_no}
train, test = torch.utils.data.random_split(
    dataset,
    [size_train, size_test],
    generator=torch.Generator().manual_seed(seed) if seed else None,
)

batch_size =  128
train_loader = DataLoader(
        train, batch_size, shuffle=True, **args
)
val_loader = (
    DataLoader(
        test, batch_size=batch_size, shuffle=False, **args
    )
    if test
    else None
)
logger = TensorBoardLogger("tb_logs", name="test")

profiler = PyTorchProfiler(with_stack=True)
trainer = Trainer(
            gpus=1,
            max_epochs=200,
            enable_model_summary=False,
            enable_progress_bar=True,
            log_every_n_steps=2,
            logger=logger,
            profiler="pytorch"
            # progress_bar_refresh_rate=20,
        )

        # Train the model
trainer.fit(model, train_loader, val_dataloaders=val_loader)