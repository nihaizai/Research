# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:47:55 2019

@author: Administrator
"""

import os,shutil
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from PIL import Image

import transforms
import numpy as np

#函数说明：test_set
#函数功能：将之前从63到124个体的GEI提取出来作为probe set和gallery set。
#参数说明：srcpath，原始的保存所有个体GEI的文件路径，eg：'E:\\GEI_B\\test_src\\'
#        probe_path, probe set的文件路径，eg：'E:\\GEI_B\\test\\probe\\'
#        gallery_path,gallery set的文件路径，eg：'E:\\GEI_B\\test\\gallery\\'
def test_set(srcpath,probe_path,gallery_path):
    probe_conds = ['nm-05','nm-06','cl-01','cl-02','bg-01','bg-02']
    gallery_conds = ['nm-01','nm-02','nm-03','nm-04']
    angles = ['000','018','036','054','072','090','108','126','144','162','180']
#    conds = ['nm-05','nm-06','cl-02','bg-02']
#    nexist_file = open('D:\\@mmg\\GaitRecognition\\PTSN\\data_pre\\test_img_json\\nexist_file.txt','a')
#    nexist_file = open('D:\\@mmg\\GaitRecognition\\PTSN\\data_pre\\train_img_json\\nexist_file.txt','a')
    nexist_file = open('E:\\GEI_B\\nexist_file.txt','a')
    print('nexist_file is open')
    dirs = os.listdir(srcpath)
    print('dirs:{}'.format(dirs))
    for i in dirs:   #probe (63,125)  train(1,63)
        idx = i
#        print('idx:{}'.format(idx))
        for angle in angles:
            for probe_cond in probe_conds:
                fname = str(idx) + '-' +probe_cond +'-'+angle + '.png'
                #print('fname:{}'.format(fname))
                src_probe = os.path.join(srcpath+i,fname)
                print('src_probe:{}'.format(src_probe))
                if not os.path.exists(src_probe):
                    print('%s not exist'%(src_probe))                   
                    nexist_file.writelines(src_probe + '\n')
                else:
                    dst_probe = os.path.join(probe_path,fname)
                    shutil.copy(src_probe,dst_probe)
                    
            for gallery_cond in gallery_conds:
                gname = str(idx) + '-' + gallery_cond +'-'+angle + '.png'
                src_gallery = os.path.join(srcpath+i,gname)
                print('src_gallery:{}'.format(src_gallery))
                if not os.path.exists(src_gallery):
                    print('%s  not exist' %(src_gallery))
                    nexist_file.writelines(src_gallery + '\n')
                else:
                    dst_gallery = os.path.join(gallery_path,gname)
                    shutil.copy(src_gallery,dst_gallery)
                    
                
    nexist_file.close()
    print("copy end!!!")


def gei_loader(path):
    with open(path,'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')



'''
   test_dset:测试数据集
'''
class test_dset(Dataset):
    def __init__(self,img_path,img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]),
                 loader = gei_loader):
        
        self.img_list = []
        self.label_list = []
        self.img_transform = img_transform
        
        path_end = os.listdir(img_path)
        path_end.sort()
        
        for file_path in path_end:
            self.img_list.append(os.path.join(img_path,file_path))
            print('img_path:{}'.format(self.img_list[-1]))
            self.label_list.append(file_path.split('.')[0])
            print('label:{}'.format(self.label_list[-1]))
        
        self.loader = loader
        
    def __getitem__(self,index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = self.loader(img_path)
        img = self.img_transform(img)
        
        return img,label
    
    def __len__(self):
        return len(self.label_list)















if __name__ == '__main__':
#    probe_set('D:\\@mmg\\GaitRecognition\\PTSN\\data_pre\\test_img_json\\gallery_clean_joints_json\\','D:\\@mmg\\GaitRecognition\\PTSN\\data_pre\\test_img_json\\probe_clean_joints_json\\')
#    probe_set('D:\\@mmg\\GaitRecognition\\PTSN\\data_pre\\train_img_json\\clean_joints_json\\','D:\\@mmg\\GaitRecognition\\PTSN\\data_pre\\train_img_json\\test_clean_joints\\')
#    test_set('E:\\GEI_B\\test_src\\','E:\\GEI_B\\test\\probe\\','E:\\GEI_B\\test\\gallery\\')
    gallery_set = test_dset('/home/mg/code/data/GEI_B/test_rst/gallery/')  
    gallery_loader = DataLoader(gallery_set,batch_size=128,shuffle=False)
    for batch_img,batch_label in gallery_loader:
        print('batch_img.shape:{}'.format(batch_img.shape))
        batch_label = np.array(batch_label)
        print('batch_label.shape:{}'.format(batch_label.shape))
        print('batch_label:{}'.format(batch_label))
        print('one batch')
    

                
    
#    file_move('D:\\@mmg\\GaitRecognition\\PTSN\\data_pre\\test_img_json\\gallery_clean_joints_json\\063-bg-01-000.json','D:\\@mmg\\GaitRecognition\\PTSN\\data_pre\\test_img_json\\probe_clean_joints_json\\')

