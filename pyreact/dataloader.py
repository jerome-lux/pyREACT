import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset


class DefaultDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        file_pattern="*.npy",
    ):

        super().__init__()
        self.data_dir = data_dir
        self.filenames = list(Path(data_dir).glob(file_pattern))
        self.transform = transform

        if len(self.filenames) == 0:
            raise ValueError(f"No files found in {data_dir} matching pattern {file_pattern}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if idx >= len(self.filenames):  # Basic bounds check
            raise IndexError(f"Index {idx} out of bounds for a dataset of size {len(self.filenames)}.")

        filepath = self.filenames[idx]

        try:
            array = np.load(filepath)

        except FileNotFoundError:  # Should not occur if __init__ was correct
            print(f"FATAL: File {filepath} not found in __getitem__. Sample index: {idx}")
            raise
        except Exception as e:
            print(f"Error loading or slicing data from {filepath} for sample index {idx}: {e}")
            raise

        array = torch.from_numpy(array).float()  # Ensure float type

        if self.transform is not None:
            try:
                array = self.transform(array)
            except Exception as e:
                print(f"Error applying transform to data from {filepath} for sample index {idx}: {e}")
                raise

        return array


