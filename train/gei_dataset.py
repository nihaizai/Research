# -*- coding: utf-8 -*-
"""
Created on Wed May  8 20:47:05 2019

@author: Administrator
"""

from torchvision.datasets import ImageFolder
import torch 
from torchvision import transforms

from PIL import Image
from skimage.measure import compare_ssim
from scipy.misc import imread



def gei_loader(path):
    with open(path,'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
    
    
if __name__ == '__main__':
    
    transform = transforms.Compose([
            transforms.ToTensor()
            ])
    trainset = ImageFolder('/home/mg/code/data/GEI_B/train/',
                           transform = transform,loader = gei_loader )
#    print('trainset[0].size:{}'.format(trainset[0][0].shape))
#    print('trainset[0].label:{}'.format(trainset[0][1]))
#    im_path = trainset.imgs[0][0]
#    print('im_path:{}'.format(im_path))
#    img = Image.open(im_path)
#    img.show()
#    img.save('/home/mg/code/data/GEI_B/1.png')
#    img_train = imread('/home/mg/code/data/GEI_B/train/001/001-bg-01-000.png')
#    img_test = imread('/home/mg/code/data/GEI_B/1.png')
#    result = compare_ssim(img_train,img_test)
#    print('result:{}'.format(result))




  
#    count = 0
#    for path,idx in trainset.imgs:
#        print('path:{}'.format(path))
#        print('idx:{}'.format(idx))
#        count += 1
#        print('count:{}'.format(count))
#    
#    print('total count:{}'.format(count))
#    print(trainset.class_to_idx)
#    print(trainset.imgs)
    
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=5, shuffle=False)
    
    for batch_idx, (data, labels) in enumerate(trainloader):
        print('batch_idx:{}'.format(batch_idx))
        print('data.shape:{}'.format(data.shape))
        print('labels:{}'.format(labels))
    
