import os
import numpy as np
import torchvision.transforms as T
from torchvision import datasets as dset
import torch as th
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def numpy_collate(batch):
    if isinstance(batch[0], (np.ndarray, th.Tensor)):
        return np.stack(batch).transpose(0, 2, 3, 1)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]

    data = np.array(batch)
    if data.ndim == 4:
        return data.transpose(0, 2, 3, 1)
    return data
    

class ImagePaths(Dataset):
    def __init__(self, path: str, transform=None, max_size=None):
        self.images = [os.path.join(path, img) for img in os.listdir(path)]
        if max_size:
            self.images = self.images[:max_size]
        self._length = len(self.images)
        self.transform = transform

    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img
    

def load_folder_data(path: str, batch_size: int, shuffle: bool = False, num_workers: int = 0, transform=None, max_size: int = None):
    train_set = ImagePaths(path, transform=transform, max_size=max_size)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate,
                        drop_last=True, num_workers=num_workers, pin_memory=False)
    return loader


def load_stl(data_dir, split, batch_size, shuffle=False, num_workers=0, transform=None):
    if transform is None:
        if split == 'train':
            transform = T.Compose([T.Resize(96),
                                   T.RandomHorizontalFlip(),
                                   T.ToTensor(),
                                   T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = T.Compose([T.Resize(96),
                                   T.ToTensor(),
                                   T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = dset.STL10(data_dir, split=split, transform=transform, download=False)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=numpy_collate, drop_last=True)
    return loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time
    from tqdm import tqdm
    path = os.path.expanduser("~/PycharmProjects/Datasets/ILSVRC2012_img_test/test")
    paths = [os.path.join(path, img) for img in os.listdir(path)]

    def check_ndim(x):
        if x.ndim == 2:
            x = th.stack([x] * 3, 0)
        elif x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        return x

    transform = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
    th_loader = load_folder_data(path, 64, transform=transform, num_workers=8)

    st = time()
    for i, img in tqdm(enumerate(th_loader), total=30):
        if i > 30:
            print(f'Torch: {time() - st:.4f}')
            plt.imshow(img[0].transpose(1, 2, 0))
            plt.show()
            break

    # st = time()
    # for i in tqdm(range(30)):
    #     img = jx_loader.sample(ids=np.arange(64) + i * 64)
    # else:
    #     print(f'Jax: {time() - st:.4f}')
    #     plt.imshow(img[0])
    #     plt.show()