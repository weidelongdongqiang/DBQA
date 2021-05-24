#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 08:32:10 2021

@author: weizhe
"""
from Model import *
from utils import *
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Your image to be assess")
parser.add_argument('--image', required=True, type=str)
args = parser.parse_args()

# Load
device=torch.device('cuda')
model=Integrate_plus(device)
model.load_state_dict(torch.load('Rank.pth'))
model.to(device)
print('Successfully load model.')

norm_mean=[0.5,0.5,0.5]
norm_std=[.5,.5,.5]
trans=transforms.Compose([transforms.TenCrop(300),
                         transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                         transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(norm_mean, norm_std)(crop) for crop in crops]))
                         ])

model.eval()
with torch.no_grad():
    img=Image.open(args.image).convert('RGB')
    X=trans(img)
    X=X.cuda()
    y=model(X)
Score=y.detach().mean().cpu().numpy()
print('%s: %.4f' % (args.image,Score))
