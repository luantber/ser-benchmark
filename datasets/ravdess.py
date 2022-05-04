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
        self.audio_filenames = self.audio_labels["file"].to_numpy()
        self.audio_emotions = self.audio_labels["emotion"].to_numpy()

        self.audio_cache = {}
        self.new_sample_rate = 8000

        # breakpoint()

    def __len__(self):
        return len(self.audio_labels)

    # def get_sr(self):
    #     audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[0, 0])
    #     wave, sr = torchaudio.load(audio_path)
    #     return sr

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_filenames[idx] )
        
        # Audio Cache load
        if audio_path in self.audio_cache:
            wave_resampled = self.audio_cache[audio_path]
            # print(audio_path,"cache")
        else:
            wave_stereo, sr = torchaudio.load(audio_path)
            wave_mono = torch.mean(wave_stereo, dim=0, keepdim=True)

            resample_transform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.new_sample_rate
            )

            wave_resampled = resample_transform(wave_mono)
            self.audio_cache[audio_path] = wave_resampled
            # print(audio_path,"load")

        # Get Label
        label = torch.tensor(self.audio_emotions[idx])
        #dimension,size,step
        unfolded = wave_resampled[0].unfold(0, self.new_sample_rate, self.new_sample_rate // 2)

        return unfolded, label
