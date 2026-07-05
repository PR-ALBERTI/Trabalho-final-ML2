import torch
from torch.utils import data
import os
import numpy as np
import torch.nn.functional as F
import h5py


# ── Dataset original (volumes 3D, pipelines COMPLETE/IND) ───

class fMRICNNcustomDataset(data.Dataset):
    def __init__(self, data_folder):
        self.path_list = []
        self.root_folder = data_folder
        for path, subdirs, files in os.walk(self.root_folder):
            for name in files:
                if name.endswith('.p'):
                    self.path_list.append(os.path.join(path, name))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = torch.load(self.path_list[index])
        x_data = np.array(sample['X'])
        y_cluster = np.array(sample['y'])
        return x_data, y_cluster


# ── Dataset AVG volumétrico (pipeline AVG 3D) ───────────────

TARGET_SHAPE_DL = (91, 109, 91)

class fMRIAVGDataset(data.Dataset):

    def __init__(self, hdf5_path: str, target_shape: tuple = TARGET_SHAPE_DL):
        self.hdf5_path    = hdf5_path
        self.target_shape = target_shape
        self._file        = None

        with h5py.File(hdf5_path, "r") as f:
            self.n        = len(f["nsd_ids"])
            self.labels   = f["labels"][:]
            self.coco_ids = f["coco_ids"][:]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

        vol = self._file["betas"][index]
        vol = np.nan_to_num(vol.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        t = torch.tensor(vol).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=self.target_shape, mode="trilinear", align_corners=False)
        X = t.squeeze(0)  # (1, 91, 109, 91)

        y = int(self.labels[index])
        return X, torch.tensor(y, dtype=torch.long)

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ── Dataset ROI (vetores flat, pipeline ROI-AVG) ────────────

class fMRIROIDataset(data.Dataset):

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self._file     = None

        with h5py.File(hdf5_path, "r") as f:
            self.n        = f["betas"].shape[0]
            self.n_voxels = f["betas"].shape[1]
            self.labels   = f["labels"][:]
            self.coco_ids = f["coco_ids"][:]
            self.nsd_ids  = f["nsd_ids"][:]

        print(f"[fMRIROIDataset] {hdf5_path}")
        print(f"  n={self.n:,} | n_voxels={self.n_voxels:,}")
        print(f"  label dist: {dict(zip(*np.unique(self.labels, return_counts=True)))}")

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

        vec = self._file["betas"][index].astype(np.float32)
        X   = torch.tensor(vec)                          # (n_voxels,)
        y   = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return X, y

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ── Augmentação p/ ROI (ruído gaussiano + dropout de voxels) ──

class AugmentedROIWrapper(data.Dataset):
    def __init__(self, base_dataset, noise_std: float = 0.05, dropout_p: float = 0.1):
        self.base_dataset = base_dataset
        self.noise_std    = noise_std
        self.dropout_p    = dropout_p

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        X, y = self.base_dataset[index]
        if self.noise_std > 0:
            X = X + torch.randn_like(X) * (self.noise_std * X.std())
        if self.dropout_p > 0:
            keep = (torch.rand_like(X) > self.dropout_p).float()
            X = X * keep
        return X, y