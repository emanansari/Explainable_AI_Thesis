import torch
import torchaudio
import os
from torch.utils.data import Dataset
import pandas as pd

class UrbanSound8K(Dataset):
    def __init__(self, datapath, device, folders_in=None, transform=None, max_length=400):
        self.csv_location = '/home3/s4317394/pytorch/UrbanSound8K.csv'
        self.audio_root = datapath
        self.device = device
        self.transform = transform.to(device)
        self.max_length = max_length  # Maximum length for padding/truncating
        self._load_data_paths(folders_in)

    
    def _load_data_paths(self, folders_in):
        df = pd.read_csv(self.csv_location)
        print("Number of rows in DataFrame:", len(df))

        if folders_in is not None:
            df = df[df['fold'].isin(folders_in)]
        print("Number of rows after filtering folders:", len(df))

        self.paths = [os.path.join(self.audio_root, f'fold{row["fold"]}', row['slice_file_name']) for _, row in df.iterrows()]
        self.labels = [int(row['classID']) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.paths)
    
    def _file_to_mfcc(self, index):
        waveform, sr = torchaudio.load(self.paths[index])
        waveform = waveform.to(self.device)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        if sr != self.transform.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.transform.sample_rate).to(self.device)
            waveform = resampler(waveform)

        mfcc = self.transform(waveform)

        # Pad or truncate the mfcc to the maximum length
        if mfcc.shape[-1] < self.max_length:
            pad_amount = self.max_length - mfcc.shape[-1]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))
        elif mfcc.shape[-1] > self.max_length:
            mfcc = mfcc[:, :, :self.max_length]

        return mfcc
    
    def __getitem__(self, index):
        return self._file_to_mfcc(index), self.labels[index]

print("done importing")


class TransformedUrbanSound8K(UrbanSound8K):
    def __init__(self, datapath, device, folders_in=None, transform=None, max_length=400):
        super(TransformedUrbanSound8K, self).__init__(datapath, device, folders_in, transform, max_length)

    def __getitem__(self, index):
        data, label = super(TransformedUrbanSound8K, self).__getitem__(index)
        data = self.transform(data)  # Apply transformation
        return data, label


