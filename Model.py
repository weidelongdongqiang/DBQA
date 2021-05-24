#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:11:50 2021

@author: weizhe
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_gdn import GDN
import torchvision.models as models

class myVgg19(nn.Module):
    def __init__(self):
      super(myVgg19, self).__init__()
      self.conv1=nn.Conv2d(3,64,3,stride=1,padding=1)
      self.actfunc=nn.ReLU()
      self.conv2=nn.Conv2d(64,64,3,stride=1,padding=1)
      self.pool=nn.MaxPool2d(2,stride=2)
      self.conv3=nn.Conv2d(64,128,3,stride=1,padding=1)
      self.conv4=nn.Conv2d(128,128,3,stride=1,padding=1)
      self.conv5=nn.Conv2d(128,256,3,stride=1,padding=1)
      self.conv6=nn.Conv2d(256,256,3,stride=1,padding=1)
      self.conv7=nn.Conv2d(256,256,3,stride=1,padding=1)
      self.conv8=nn.Conv2d(256,256,3,stride=1,padding=1)

    def forward(self, X):
        # 1
        hidden=self.conv1(X)
        hidden=self.actfunc(hidden)
        hidden=self.conv2(hidden)
        hidden=self.actfunc(hidden)
        out1=self.pool(hidden)
        # 2
        hidden=self.conv3(out1)
        hidden=self.actfunc(hidden)
        hidden=self.conv4(hidden)
        hidden=self.actfunc(hidden)
        out2=self.pool(hidden)
        # 3
        hidden=self.actfunc(self.conv5(out2))
        hidden=self.actfunc(self.conv6(hidden))
        hidden=self.actfunc(self.conv7(hidden))
        hidden=self.actfunc(self.conv8(hidden))
        out3=self.pool(hidden)
        return out1, out2, out3

# input: 256,256,3
class branch256(nn.Module):
    def __init__(self):
        super(branch256, self).__init__()
        device = torch.device('cuda')
        self.conv1=nn.Conv2d(3,32, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(32,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32, 64, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(64,device)
        self.conv3=nn.Conv2d(64, 128, (5,5), stride=(2,2), padding=(2,2))
        self.gdn3=GDN(128,device)
        self.conv4=nn.Conv2d(128, 256, (3,3), stride=(1,1), padding=(0,0))
        self.gdn4=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        # 4
        hidden = self.conv4(hidden)
        hidden = self.gdn4(hidden)
        hidden = self.pool(hidden)
        return hidden

# MEON128 strenthen
# input: 128,128,64
class branch128(nn.Module):
    def __init__(self):
        super(branch128, self).__init__()
        device = torch.device('cuda')
        self.conv1=nn.Conv2d(64,128, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(128,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(128, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(256,device)
        self.conv3=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(1,1))
        self.gdn3=GDN(256,device)
        self.conv4=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(0,0))
        self.gdn4=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        # 4
        hidden = self.conv4(hidden)
        hidden = self.gdn4(hidden)
        hidden = self.pool(hidden)
        return hidden

# input: 64,64,128
class branch64(nn.Module):
    def __init__(self):
        super(branch64, self).__init__()
        device = torch.device('cuda')
        self.pool=nn.MaxPool2d(2)
        self.conv1=nn.Conv2d(128, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(256,device)
        self.conv2=nn.Conv2d(256, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(256,device)
        self.conv3=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(0,0))
        self.gdn3=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        return hidden

# input: 32,32,256
class branch32(nn.Module):
    def __init__(self):
        super(branch32, self).__init__()
        device = torch.device('cuda')
        self.pool=nn.MaxPool2d(2)
        self.conv1=nn.Conv2d(256, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(256,device)
        self.conv2=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(1,1))
        self.gdn2=GDN(256,device)
        self.conv3=nn.Conv2d(256, 256, (3,3), stride=(1,1), padding=(0,0))
        self.gdn3=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        return hidden

# integrated CNN
# with vgg19 pretrained
class Integrate(nn.Module):
    def __init__(self):
        super(Integrate, self).__init__()
        self.myVgg19=myVgg19()
        self.branch256=branch256()
        self.branch128=branch128()
        self.branch64=branch64()
        self.branch32=branch32()
        self.classifier=nn.Sequential(
            nn.Conv2d(1024,256,(1,1),stride=(1,1),padding=0),
            nn.Dropout(0.5),
            nn.Conv2d(256, 1, (1,1), stride=(1,1), padding=(0,0))
            )
        self.Att=AttentionModule(3,1,(256,256),(128,128),(64,64))
    
    def forward(self, X):
        B128,B64,B32=self.myVgg19(X)
        M=self.Att(X)
        # branch1
        X_b1=(1+M)*X
        feature256=self.branch256(X_b1)
        # branch2
        M2=nn.functional.interpolate(M,128,mode='bilinear',align_corners=True)
        X_b2=(1+M2)*B128
        feature128=self.branch128(X_b2)
        # branch3
        M3=nn.functional.interpolate(M,64,mode='bilinear',align_corners=True)
        X_b3=(1+M3)*B64
        feature64=self.branch64(X_b3)
        # branch4
        M4=nn.functional.interpolate(M,32,mode='bilinear',align_corners=True)
        X_b4=(1+M4)*B32
        feature32=self.branch32(X_b4)
        Integ=torch.cat((feature256,feature128,feature64,feature32), 1)
        hidden=self.classifier(Integ)
        return hidden.view(hidden.size(0))

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        # self.bn1 = nn.BatchNorm2d(input_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, 64, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        
    def forward(self, x):
        if (self.input_channels != self.output_channels):
            residual = self.conv4(x)
        else:
            residual=x
        # out = self.bn1(x)
        # out1 = self.relu(out)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class AttentionModule(nn.Module):

    def __init__(self, in_channels, out_channels, size1, size2, size3):
        super(AttentionModule, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(out_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(out_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = ResidualBlock(out_channels, out_channels)

        self.skip2_connection_residual_block = ResidualBlock(out_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.softmax4_blocks = ResidualBlock(out_channels, out_channels)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax5_blocks = ResidualBlock(out_channels, out_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(out_channels, out_channels)

    def forward(self, X):
        X = self.first_residual_blocks(X)
        out_mpool1 = self.mpool1(X)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        #
        out_interp3 = self.interpolation3(out_softmax3)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)

        return out_softmax6

# Regularization loss
class Regularization(nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight()
        self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self):
        # self.weight_list=self.get_weight(self.model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
    
    def get_weight(self):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
    
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")

class rank_loss1(nn.Module):
    def __init__(self):
        super(rank_loss1, self).__init__()
        return
    
    def forward(self, Y1, Y2, delta):
        loss=torch.clamp(Y2-Y1+delta, min=0)
        return loss

class Joint_loss2(nn.Module):
    def __init__(self):
        super(Joint_loss2, self).__init__()
        return
    
    def forward(self, Y1, Y2, Y1_, Y2_, lamda, delta):
        loss1=F.smooth_l1_loss(Y1_, Y1)+F.smooth_l1_loss(Y2_, Y2)
        loss2=torch.clamp(Y2_-Y1_+delta, min=0)
        return loss1+lamda*loss2

# input: 300,300,3
class branch300(nn.Module):
    def __init__(self, device):
        super(branch300, self).__init__()
        self.conv1=nn.Conv2d(3,32, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(32,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32, 64, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(64,device)
        self.conv3=nn.Conv2d(64, 128, (3,3), stride=(2,2), padding=(1,1))
        self.gdn3=GDN(128,device)
        self.conv4=nn.Conv2d(128, 256, (3,3), stride=2, padding=(1,1))
        self.gdn4=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        # 4
        hidden = self.conv4(hidden)
        hidden = self.gdn4(hidden)
        hidden = self.pool(hidden)
        return hidden

# input: 75,75,64
class branch75(nn.Module):
    def __init__(self, device):
        super(branch75, self).__init__()
        self.conv1=nn.Conv2d(64,128, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(128,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(128, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn2=GDN(256,device)
        self.conv3=nn.Conv2d(256, 256, (3,3), stride=(2,2), padding=(1,1))
        self.gdn3=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        return hidden

# input: 37,37,128
class branch37(nn.Module):
    def __init__(self, device):
        super(branch37, self).__init__()
        self.conv1=nn.Conv2d(128, 256, (5,5), stride=(2,2), padding=(2,2))
        self.gdn1=GDN(256,device)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(256, 256, (3,3), stride=1, padding=1)
        self.gdn2=GDN(256,device)
        self.conv3=nn.Conv2d(256, 256, (3,3), stride=2, padding=(1,1))
        self.gdn3=GDN(256,device)
        
    def forward(self, X):
        # 1
        hidden = self.conv1(X)
        hidden = self.gdn1(hidden)
        hidden = self.pool(hidden)
        # 2
        hidden = self.conv2(hidden)
        hidden = self.gdn2(hidden)
        hidden = self.pool(hidden)
        # 3
        hidden = self.conv3(hidden)
        hidden = self.gdn3(hidden)
        hidden = self.pool(hidden)
        return hidden

# integrated CNN++
# with resnet34 pretrained
# input: 300*300*3
class Integrate_plus(nn.Module):
    def __init__(self, device):
        super(Integrate_plus, self).__init__()
        self.myResNet34=models.resnet34(pretrained=True)
        self.base_layers = list(self.myResNet34.children())
        self.layer1=nn.Sequential(*self.base_layers[:4])
        self.layer2=self.base_layers[4]
        self.layer3=self.base_layers[5]
        self.branch300=branch300(device)
        self.branch75_0=branch75(device)
        self.branch75_1=branch75(device)
        self.branch37=branch37(device)
        self.classifier=nn.Sequential(
            nn.Conv2d(1024,256,(1,1),stride=(1,1),padding=0),
            nn.Dropout(0.5),
            nn.Conv2d(256, 1, (1,1), stride=(1,1), padding=(0,0))
            )
    
    def forward(self, X):
        feature1=self.layer1(X)
        feature2=self.layer2(feature1)
        feature3=self.layer3(feature2)
        o300=self.branch300(X)
        o75_0=self.branch75_0(feature1)
        o75_1=self.branch75_1(feature2)
        o37=self.branch37(feature3)
        Integ=torch.cat((o300,o75_0,o75_1,o37), 1)
        hidden=self.classifier(Integ)
        return hidden.view(hidden.size(0))
