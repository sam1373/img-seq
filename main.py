import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from train import Trainer

from model import *

import matplotlib.pyplot as plt

#from data import CelebaDataset

import numpy as np




batch_size = 128

sep = 8

#model = ImgAttendModel(side_len=16, kernels=3, channels=128, blocks_rep=4, conv_rep=4)
model = PixelCNN(side_len=28, kernels=7, in_channels=3, channels=128, out_channels=200)
model = nn.DataParallel(model)

total = 0
for p in model.parameters():
  total += np.prod(p.shape)
  print(p.shape)
  
print(total)
#input()

rescaling = lambda x : (x - .5) * 2.

custom_transform = transforms.Compose([transforms.ToTensor()])

dataset = "MNIST"

if dataset == "CelebA":
  train_dataset = CelebaDataset(txt_path='/home/samuel/Data/CelebAligned/list_attr_celeba.txt',
                                img_dir='/home/samuel/Data/CelebAligned/',
                                transform=custom_transform, in_size=model.side_len)

  trainloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
elif dataset == "MNIST":

  trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/skriman2/MNIST/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                 #transforms.Resize((16, 16)),
                                 transforms.ToTensor(),
                                 #rescaling
                                 #transforms
                                 #transforms.ColorJitter(0.1, 0.1, 0.1),
                                 #transforms.Normalize(
                                 #  (0.5,), (0.5,))
                               ])),
    batch_size=batch_size, shuffle=True, num_workers=0)


examples = enumerate(trainloader)



_, (sample, _) = next(examples)
print(sample.shape)


#plt.imshow(sample[0].permute(1, 2, 0))
#plt.show()

#input()

#sample = torch.unsqueeze(sample,0)
sample = sample.cuda()

trainer = Trainer(model)

trainer.model.eval()

#trainer.umap_codes(0, trainloader)
#trainer.sample_frames(0)
#trainer.recon_frame(0, sample)

trainer.train_model(trainloader, test_every_x=1, epochs=100)

