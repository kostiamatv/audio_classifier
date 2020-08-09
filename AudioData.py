import os
import numpy as np
import librosa
from scipy.io.wavfile import read
from torch.utils.data import Dataset
from tqdm import tqdm


class AudioData(Dataset):
    def __init__(self, base, df, path_col, silent=False):
        self.df = df
        self.data = []
        self.labels = df["target"]
        for ind in tqdm(range(len(df)), desc="Processing data: ",
                        leave=False,
                        disable=silent):
            row = df.iloc[ind]
            file_path = os.path.join(base, row[path_col])
            self.data.append(
                self.preprocess_file(file_path)[np.newaxis, ...])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @staticmethod
    def preprocess_file(file_path):
        window_size = 0.02
        window_stride = 0.01
        sample_rate = 16000

        n_fft = int(sample_rate * (window_size + 1e-8))
        win_length = n_fft
        hop_length = int(sample_rate * (window_stride + 1e-8))

        _, wav = read(file_path)
        abs_max = np.abs(wav).max()
        wav = wav.astype('float32')
        if abs_max > 0:
            wav *= 1 / abs_max

        res = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length)
        res, _ = librosa.magphase(res)
        return res
