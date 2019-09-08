import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from skimage.transform import resize


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, txt_path, img_dir, transform=None, in_size=64):
    
        df = pd.read_csv(txt_path, sep=",", index_col=0)
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.img_names = df.index.values
        #self.y = df['Male'].values
        self.transform = transform
        self.in_size = in_size

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        img = np.asarray(img, dtype=np.uint8)

        #print(img.max())

        w = img.shape[0]
        h = img.shape[1]
        w_b = int(w * 0.2)
        h_b = int(h * 0.2)

        img = img[w_b:-w_b, h_b:-h_b]

        img = resize(img, (self.in_size, self.in_size, 3))
        
        if self.transform is not None:
            img = self.transform(img)

        img = img.float()
        
        #label = self.y[index]
        return img, [0.]

    def __len__(self):
        return len(self.img_names)#self.y.shape[0]







if __name__ == '__main__':

    custom_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CelebaDataset(txt_path='/home/samuel/Data/CelebAligned/list_attr_celeba.txt',
                                  img_dir='/home/samuel/Data/CelebAligned/',
                                  transform=custom_transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=128,
                              shuffle=True,
                              num_workers=1)


    num_epochs = 2
    for epoch in range(num_epochs):

        for batch_idx, (x, y) in enumerate(train_loader):
            
            print('Epoch:', epoch+1, end='')
            print(' | Batch index:', batch_idx, end='')
            print(' | Batch size:', y.size()[0])
            
            x = x.cuda()
            y = y.cuda()
           
           
            one_image = x[0].permute(1, 2, 0)
            one_image.shape

            plt.imshow(one_image.to(torch.device('cpu')).squeeze())
            plt.show()