from dataclasses import dataclass
import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
from datasets import utils


@dataclass
class Ravdess(Dataset):

    annotation_file: str
    audio_dir: str

    def __post_init__(self):
        self.audio_labels = pd.read_csv(self.annotation_file)
        self.audio_cache = {}

    # def __init__(self, annotations_file: str, audio_dir:str, transform=None):
    #     self.annotation_file = annotations_file
    #     self.audio_labels = pd.read_csv(self.annotation_file)
    #     self.audio_dir = audio_dir
    #     self.transform = transform

    #     self.spec_cache = {}

    # def set_transform(self, transform):
    #     self.transform = transform

    def __len__(self):
        return len(self.audio_labels)

    def get_sr(self):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[0, 0])
        wave, sr = torchaudio.load(audio_path)
        return sr

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 0])

        # wave, sr = torchaudio.load(audio_path)
        # wave = torch.mean(wave, dim=0, keepdim=True)

        if audio_path in self.audio_cache:
            wave = self.audio_cache[audio_path]
            # wave = torch.mean(wave, dim=0, keepdim=True)[0]
        else:
            wave, sr = torchaudio.load(audio_path)
            wave = torch.mean(wave, dim=0, keepdim=True)

            new_sample_rate = 8000
            transform = torchaudio.transforms.Resample(orig_freq=self.get_sr(), new_freq=new_sample_rate)
            wave = transform(wave)

            self.audio_cache[audio_path] = wave

        #     # only supporting over Spectrogram
        #     if self.transform:
        #         if audio_path in self.spec_cache:
        #             wave = self.spec_cache[audio_path]
        #         else:
        #             spec = utils.melspectrogram(wave, sr)
        #             wave = utils.powertodb(spec, sr)
        #             self.spec_cache[audio_path] = wave

        label = torch.tensor(self.audio_labels.iloc[idx, 1])
        # print(type(label))

        #     if self.transform:
        #         for t in self.transform:
        #             wave = t(wave, sr)

        return wave, label
