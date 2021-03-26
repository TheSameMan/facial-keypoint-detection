"Custom dataset and transform classes"
import os
import pandas as pd
import torch as tr
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize, CenterCrop, Normalize, ToTensor


class KeypointDataset(tr.utils.data.Dataset):
    """Dataset of facial images with keypoints
    of the facial contour
    """
    def __init__(self, root, transform=None, loader=default_loader):
        super().__init__()
        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root

        self.transform = transform
        self.loader = loader
        self.kpts = self._find_kpts()

    def _find_kpts(self):
        with os.scandir(self.root) as itr:
            kpts_file = None
            for entry in itr:
                _, ext = os.path.splitext(entry.name)
                if ext == '.csv':
                    kpts_file = os.path.join(self.root, entry.name)
                    kpts = pd.read_csv(kpts_file)

            if kpts_file is None:
                raise FileNotFoundError('Keypoints file not found')

        return kpts

    def __getitem__(self, index):
        names = self.kpts.columns[0]
        path = os.path.join(self.root, self.kpts[names].iloc[index])

        points = self.kpts.columns[1:]
        target = self.kpts[points].iloc[index].to_numpy().reshape(-1, 2)

        sample = self.loader(path), target
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.kpts)


class CenterCropped(CenterCrop):
    """Analogue of the CenterCrop class for a sample
    containing an image with keypoints
    """

    def __init__(self, size):
        super().__init__(size)

    def forward(self, sample):
        img, kpts = sample
        if not isinstance(self.size, (tuple, list)):
            self.size = (self.size, self.size)

        if self.size != img.size:
            if img.size[1] != img.size[0] or self.size[1] != self.size[0]:
                k = (img.size[1] - self.size[1]) / (img.size[0] - self.size[0])
            else:
                k = 1

            kpts[:, 0] -= (img.size[0] - self.size[0])*k / 2
            kpts[:, 1] -= (img.size[1] - self.size[1]) / (2*k)

        return super().forward(img), kpts


class Resized(Resize):
    """Analogue of the Resize class for a sample
    containing an image with keypoints
    """
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, interpolation)

    def forward(self, sample):
        img, kpts = sample
        if not isinstance(self.size, (tuple, list)):
            self.size = (self.size, self.size)
        return super().forward(img), kpts * self.size / img.size


class Normalized(Normalize):
    """Analogue of the Normalize class for a sample
    containing an image with keypoints
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 kpts_mean=(0, 0), kpts_std=(1, 1), inplace=False):
        super().__init__(mean, std, inplace)
        self.kpts_mean = tr.as_tensor(kpts_mean)
        self.kpts_std = tr.as_tensor(kpts_std)

    def forward(self, sample):
        img, kpts = sample
        if isinstance(img, tr.Tensor) and isinstance(kpts, tr.Tensor):
            return (super().forward(img),
                    kpts.sub_(self.kpts_mean).div_(self.kpts_std))
        raise TypeError('Normalization requires tensor images and keypoints')


class ToTensored(ToTensor):
    """Analogue of the ToTensor class for a sample
    containing an image with keypoints
    """
    def __call__(self, sample):
        img, kpts = sample
        return super().__call__(img), tr.as_tensor(kpts)
