import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


label_map = {
    'Forehead - pull hairline': 0,
    'Neck - pinch skin': 1,
    'Text on phone': 2,
    'Neck - scratch': 3,
    'Forehead - scratch': 4,
    'Eyelash - pull hair': 5,
    'Above ear - pull hair': 6,
    'Eyebrow - pull hair': 7,
    'Cheek - pinch skin': 8,
    'Wave hello': 9,
    'Write name in air': 10,
    'Pull air toward your face': 11,
    'Feel around in tray and pull out an object': 12,
    'Write name on leg': 13,
    'Pinch knee/leg skin': 14,
    'Scratch knee/leg skin': 15,
    'Drink from bottle/cup': 16,
    'Glasses on/off': 17
 }
good_labels = [2, 9, 10, 11, 12, 13, 14, 15, 16, 17]
bad_labels = [0, 1, 3, 4, 5, 6, 7, 8]


class MaskedStandardScaler:
    def __init__(self, eps: float = 1e-8, mask_fill_value: float = 0):
        self.eps = eps
        self.means = None
        self.stds = None
        self.X_cols = None
        self.mask_cols = None
        self.mask_fill_value = mask_fill_value

    def create_mask(self, df: pd.DataFrame):
        mask = pd.DataFrame(index=df.index, columns=self.X_cols)
        if self.mask_cols is None:
            mask.loc[:, :] = True
            return mask
        
        for X_col, mask_col in zip(self.X_cols, self.mask_cols):
            if mask_col is None:
                mask[X_col] = df[X_col].notna()
            else:
                mask[X_col] = df[mask_col].astype(bool)
        return mask

    def fit(self, df: pd.DataFrame, X_cols: list[str], mask_cols: list[str] = None):
        self.X_cols = X_cols
        self.mask_cols = mask_cols

        X = df[X_cols].to_numpy(dtype=np.float32)
        mask = self.create_mask(df).to_numpy(dtype=bool)

        self.means = np.nanmean(np.where(mask, X, np.nan), axis=0)
        self.stds = np.nanstd(np.where(mask, X, np.nan), axis=0) + self.eps

    def transform(self, df: pd.DataFrame, inplace: bool = True):
        if not inplace:
            df = df.copy()

        X = df[self.X_cols].to_numpy(dtype=np.float32)
        mask = self.create_mask(df).to_numpy(dtype=bool)

        X_std = (X - self.means) / self.stds
        X_std[~mask] = self.mask_fill_value

        df[self.X_cols] = X_std
        return df
    
    def fit_transform(self, df: pd.DataFrame, X_cols: list[str], mask_cols: list[str] = None, inplace: bool = True):
        self.fit(df, X_cols, mask_cols)
        return self.transform(df, inplace=inplace)
    

class CMI_Dataset(Dataset):
    def __init__(self, labels, labels2, demo, feature_imu_sequence, feature_1d_sequence, feature_2d_tof_sequence, feature_2d_itof_sequence):
        self.labels = labels
        self.labels2 = labels2
        self.demo = demo
        self.feature_imu_seq = feature_imu_sequence
        self.feature_1d_seq = feature_1d_sequence
        self.feature_2d_tof_seq = feature_2d_tof_sequence
        self.feature_2d_itof_seq = feature_2d_itof_sequence

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'label': self.labels[idx],
            'label2': self.labels2[idx],
            'demo': self.demo[idx],
            'seq_imu': self.feature_imu_seq[idx],
            'seq_1d': self.feature_1d_seq[idx],
            'seq_2d_tof': self.feature_2d_tof_seq[idx],
            'seq_2d_itof': self.feature_2d_itof_seq[idx],
        }

    @staticmethod
    def collate_fn(batch):
        labels = torch.tensor([item['label'] for item in batch])
        labels2 = torch.tensor([item['label2'] for item in batch])
        demo = torch.stack([item['demo'] for item in batch])

        seq_imu_list = [item['seq_imu'] for item in batch]  # [B, T, 7]
        seq_1d_list = [item['seq_1d'] for item in batch]    # [B, T, 12]
        seq_2d_tof_list = [item['seq_2d_tof'] for item in batch]    # [B, T, 5, 8, 8]
        seq_2d_itof_list = [item['seq_2d_itof'] for item in batch]  # [B, T, 5, 8, 8]

        lengths = torch.tensor([t.size(0) for t in seq_imu_list])
        max_len = lengths.max()
        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        # Pad 1D
        seq_imu = pad_sequence(seq_imu_list, batch_first=True)    # [B, T, D]
        seq_1d = pad_sequence(seq_1d_list, batch_first=True)      # [B, T, D]

        # Pad 2D
        seq_2d_tof = pad_sequence(seq_2d_tof_list, batch_first=True)    # [B, T, 5, 8, 8]
        seq_2d_itof = pad_sequence(seq_2d_itof_list, batch_first=True)  # [B, T, 5, 8, 8]

        return {
            'labels': labels,
            'labels2': labels2,
            'demo': demo,
            'seq_imu': seq_imu,
            'seq_1d': seq_1d,
            'seq_2d_tof': seq_2d_tof,
            'seq_2d_itof': seq_2d_itof,
            'mask': mask,
            'lengths': lengths,
        }