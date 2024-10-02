import numpy as np
import cv2
from torch.utils.data import Dataset
import os


class FCGClassificationDataset(Dataset):
    def __init__(self, root_dir, list_data, max_sequence, transform=None):
        self.root = root_dir
        self.list = list_data
        self.max_sequence = max_sequence
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        signal, labels, fn = self.load_data(idx)
        data_loader = {'signal': signal, 'label': labels, 'fn': fn}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_data(self, idx):
        signal_dir = os.path.join(self.root, self.list[idx])
        annot_dir = os.path.join(self.root, self.list[idx][:-4] + '.txt')
        fn = self.list[idx]
        signal = np.load(signal_dir)
        signal = np.expand_dims(signal, axis=0)
        description = [line.rstrip('\n') for line in open(annot_dir)][0]
        labels = []
        components = description.split(" ")
        for idx, component in enumerate(components):
            labels.append(component)
        labels = np.array(labels).astype(np.float32)
        return signal, labels, fn
