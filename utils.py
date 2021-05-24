#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 08:44:46 2020

@author: weizhe
"""
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os,cv2
import random as rd

# For deblurQA dataset
'''
def gen_classify(txt, size):
    with open(txt) as fo:
        lines=fo.readlines()
    while True:
        X=list()
        Y=list()
        choic=rd.sample(lines,size)
        for l in choic:
            l=l.strip('\n')
            fName,cla_s = l.split(' ')
            x=cv2.imread(fName)/127.5-1
            X.append(x)
            Y.append(int(cla_s))
            Y.append(1)
        X=np.array(X)
        X=np.transpose(X,[0,3,1,2])
        Y=np.array(Y)
        yield X,Y
'''
class deblurSet(Dataset):
    def __init__(self, txt, transform=None, folder=None):
        super(deblurSet, self).__init__()
        with open(txt) as fo:
            lines=fo.readlines()
        self.Names=[]
        self.Labels=[]
        self.transform=transform
        self.folder=folder
        
        for line in lines:
            l=line.strip('\n')
            fName,Lab_s = l.split(' ')
            self.Names.append(fName)
            self.Labels.append(Lab_s)
        
    def __getitem__(self, index):
        fName=self.Names[index]
        Lab=float(self.Labels[index])
        if self.folder:
            img=Image.open(os.path.join(self.folder,fName)).convert('RGB')
        else:
            img=Image.open(fName).convert('RGB')
        if self.transform is not None:
            img=self.transform(img)
        return img, Lab
    
    def __len__(self):
        return len(self.Names)

# Gen [h,w,85] numpy result
def gen_multiscale(txt, folder=None):
    # Multi-scale data (Method2)
    size=[256,128,64,32]
    with open(txt) as fo:
        Lines=fo.readlines()
    while True:
        result=[]
        line=rd.sample(Lines,1)
        l=line[0].strip('\n')
        fName,Lab_s = l.split(' ')
        Lab=float(Lab_s)
        if folder:
            raw=cv2.imread(os.path.join(folder,fName))
        else:
            raw=cv2.imread(fName)
        raw=raw[:,:,::-1].astype(np.float32)/127.5-1 # BGR2RGB
        # random flip
        RD=rd.random()
        if 0.333<RD<0.667:
            img=cv2.flip(raw, 1)
        elif RD>=0.667:
            img=cv2.flip(raw, 0)
        else:
            img=raw.copy()
        H,W,_ = img.shape
        # adaptive size cut
        for i in range(len(size)):
            cut_y=H//(2**i)
            cut_x=W//(2**i)
            for y_int in range(2**i):
                y_low=y_int*cut_y
                y_high=(y_int+1)*cut_y-size[i]
                start_y=np.random.randint(y_low, y_high)
                for x_int in range(2**i):
                    x_low=x_int*cut_x
                    x_high=(x_int+1)*cut_x-size[i]
                    start_x=np.random.randint(x_low, x_high)
                    crop=img[start_y:start_y+size[i],start_x:start_x+size[i],:].copy()
                    crop=np.expand_dims(crop, 0)
                    crop=np.transpose(crop, [0,3,1,2])
                    result.append(torch.Tensor(crop))
        yield result, torch.Tensor([Lab])

class deblurSet1(Dataset):
    def __init__(self, txt, transform=None, folder=None):
        super(deblurSet1, self).__init__()
        with open(txt) as fo:
            lines=fo.readlines()
        self.Pairs=[]
        self.transform=transform
        self.folder=folder
        
        for line in lines:
            l=line.strip('\n')
            self.Pairs.append(l)
        
    def __getitem__(self, index):
        Pair=self.Pairs[index]
        fName1,fName2=Pair.split(' ')
        if self.folder:
            img1=Image.open(os.path.join(self.folder,fName1)).convert('RGB')
            img2=Image.open(os.path.join(self.folder,fName2)).convert('RGB')
        else:
            img1=Image.open(fName1).convert('RGB')
            img2=Image.open(fName2).convert('RGB')
        if self.transform is not None:
            img1=self.transform(img1)
            img2=self.transform(img2)
        return img1, img2
    
    def __len__(self):
        return len(self.Pairs)

if __name__=="__main__":
    from matplotlib import pyplot as plt
    '''
    trans=transforms.Compose([transforms.RandomCrop(256),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5,0.5,0.5],std=[.5,.5,.5]),
                             ])
    mySet=deblurSet('/media2/Data/deblurQA/Step2_train.txt', transform=trans,
                    folder='/media2/Data/deblurQA/deblur')
    loader=DataLoader(mySet, 4, shuffle=True)
    dataIter=iter(loader)
    X,Y = dataIter.__next__()
    plt.imshow(X[2].permute(1,2,0))
    plt.show()
    print(Y[2])
    '''
    dataGener=gen_multiscale('/media2/Data/deblurQA/Step2_train.txt',folder='/media2/Data/deblurQA/deblur')
    X,Y = dataGener.__next__()
    print(Y)
    plt.imshow(X[0][0].permute(1,2,0))
    plt.show()
    plt.imshow(X[1][0].permute(1,2,0))
    plt.show()
