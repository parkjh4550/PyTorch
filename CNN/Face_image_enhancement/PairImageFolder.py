from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import numpy as np

class DownsizedPairImageFolder(ImageFolder):
    def __init__(self, root, transform=None, large_size=128, small_size=32, **kwds):
        super().__init__(root, transform=transform, **kwds)
        self.large_resizer = transforms.Resize(large_size)
        self.small_resizer = transforms.Resize(small_size)

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        #img = torch.from_numpy(np.asarray(img))
        #print('type img : ', type(img) )
        # 128x128 -> 32x32
        large_img = self.large_resizer(img)
        small_img = self.small_resizer(img)

        #print('type large img : ', type(large_img))
        #other transformation
        if self.transform is not None:
            large_img = self.transform(large_img)
            small_img = self.transform(small_img)
        #print('type after transform : ', type(large_img))
        return small_img, large_img