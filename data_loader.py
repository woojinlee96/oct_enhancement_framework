import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch
from scipy import io
import random
from fringe_fft import fringe_FFT
from fringe_stft import fringe_STFT
import matplotlib.pyplot as plt

'''
a line size : 1300x1024 -> noverlap = 550
a line size : 850x1024 -> noverlap = 580

'''

class DatasFolder(Dataset):
    def __init__(self, root, phase):
        super(DatasFolder, self).__init__()
        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.phase_folder = phase
        self.root = root
        self.STFT_1300 = fringe_STFT(window='hann', nperseg=650, noverlap=550, nfft=2048, boundary='zeros', padded=True, axis=1)
        self.STFT_850 = fringe_STFT(window='hann', nperseg=425, noverlap=355, nfft=2048, boundary='zeros', padded=True, axis=1)

        self.FFT = fringe_FFT(nfft=2048)

        self.file_paths = os.listdir(os.path.join(self.root, "data", self.phase_folder))
        self.file_paths.sort()

        self.target_paths = os.listdir(os.path.join(self.root, "ground_truth", self.phase_folder))
        self.target_paths.sort()

        assert len(self.file_paths) == len(self.target_paths)

        self.data_paths = []
        for i in range(len(self.file_paths)):
            data_path = (os.path.join(self.root, "data", self.phase_folder, self.file_paths[i]),
                         os.path.join(self.root, "ground_truth", self.phase_folder, self.target_paths[i]))
            self.data_paths.append(data_path)

    def __getitem__(self, index):
        file_path, target_path = self.data_paths[index]  # Random index
        data_fringe = io.loadmat(file_path)
        target_fringe = io.loadmat(target_path)

        data_fringe = np.array(data_fringe['data'])
        target_fringe = np.array(target_fringe['ground_truth'])

        center_v = 105.0
        max_v = 25.0

        center_sv = 35.0
        max_sv = 35.0

        # random A-line select
        if self.phase == 'train' or self.phase =='val':
            idx = random.randrange(0, np.shape(data_fringe)[1]-3)
            data_fringe = np.expand_dims(data_fringe[:, idx:idx+2], axis=0)
            target_fringe = np.expand_dims(target_fringe[:,idx:idx+2], axis=0)

            if np.shape(data_fringe)[1] == 1300:
                data_stft = self.STFT_1300.transform(data_fringe)
            elif np.shape(data_fringe)[1] == 850:
                data_stft = self.STFT_850.transform(data_fringe)

            data = self.FFT.transform(data_fringe, phase='data').permute(2, 1, 0)
            target = self.FFT.transform(target_fringe, phase='target').permute(2, 1, 0)
            data_stft = data_stft.squeeze().permute(1,0,2)

            sh_idx1 = torch.argmax(torch.mean(target[:,0:799,:], dim=0))
            sh_idx2 = torch.argmax(torch.mean(target[:,sh_idx1 + 20:sh_idx1 + 120,:], dim=0))

            data_stft = data_stft[:, sh_idx1 + sh_idx2 + 49 : sh_idx1 + sh_idx2 + 49 + 1024, :]
            data = data[:, sh_idx1 + sh_idx2 + 49 : sh_idx1 + sh_idx2 + 49 + 1024, :]
            target = target[:, sh_idx1 + sh_idx2 + 49 : sh_idx1 + sh_idx2 + 49 + 1024, :]

            if random.random() > 0.5:
                data_stft = torch.flip(data_stft, dims=[1])
                data = torch.flip(data, dims=[1])
                target = torch.flip(target, dims=[1])

            if random.random() < 0.5:
                random_center = random.randrange(-5, 5)
                center_v = center_v + random_center
                center_sv = center_sv + random_center

            if random.random() < 0.5:
                random_max = random.randrange(-3, 3)
                max_v = max_v + random_max
                max_sv = max_sv + random_max

            data_stft = (data_stft - center_sv) / max_sv
            data = (data - center_v) / max_v
            target = (target - center_v) / max_v


        elif self.phase == 'test':
            data_fringe = data_fringe.swapaxes(0,1)
            target_fringe = target_fringe.swapaxes(0, 1)
            if np.shape(data_fringe)[1] == 1300:
                data_stft = self.STFT_1300.transform(data_fringe)
            elif np.shape(data_fringe)[1] == 850:
                data_stft = self.STFT_850.transform(data_fringe)
            data = self.FFT.transform(data_fringe, phase='data')

            target = self.FFT.transform(target_fringe, phase='target')
            data_stft = torch.reshape(data_stft, [512,2,2048,14])
            data = torch.reshape(data, [512, 2, 2048, 1])
            target = torch.reshape(target, [512, 2, 2048, 1])

            # crop_center = 1024
            # data_stft = data_stft[:, :,(crop_center - 1) - 649 : crop_center + 650, :]
            # data = data[:, :, (crop_center - 1) - 649: crop_center + 650, :]
            # target = target[:, :, (crop_center - 1) - 649: crop_center + 650, :]

            data_stft = (data_stft - center_sv) / max_sv
            data = (data - center_v) / max_v
            target = (target - center_v) / max_v

        return data_stft, data, target

    def __len__(self):
        """Returns the total number of images."""
        return len(self.data_paths)


def get_loader(dataset_path, phase="train", shuffle=True, batch_size=1, num_workers=2):
    """Builds and returns Dataloader."""

    dataset = DatasFolder(root=dataset_path, phase=phase)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader