import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from train import Trainer

from model import *

import matplotlib.pyplot as plt

from data import CelebaDataset

import numpy as np




batch_size = 128

sep = 8

#model = ImgAttendModel(side_len=16, kernels=3, channels=128, blocks_rep=4, conv_rep=4)
#model = PixelCNN(side_len=16, kernels=5, in_channels=3, channels=128, out_channels=200)
model = PixelCNNProg(side_len=28, kernels=5, in_channels=3, channels=128, out_channels=200, total_attn=9)
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

local = True


if dataset == "CelebA":
  train_dataset = CelebaDataset(txt_path='/home/samuel/Data/CelebAligned/list_attr_celeba.txt',
                                img_dir='/home/samuel/Data/CelebAligned/',
                                transform=custom_transform, in_size=model.module.side_len)

  trainloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
elif dataset == "MNIST":


  data_path = "MNIST/"

  if not local:
    data_path = "/data/skriman2/MNIST/"

  trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_path, train=True, download=True,
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


sample_path = "sample"

if not local:
  sample_path = "/data/skriman2/sample"

trainer = Trainer(model, sample_path=sample_path)

trainer.model.eval()

trainer.sample_frames(0, level=3)

trainer.train_model(trainloader, test_every_x=1, epochs=200)

