# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:33:43 2019

@author: Administrator
"""
from __future__ import division

import sys
sys.path.append('')

import  models
import argparse
from test_datasets import test_dset
from torch.utils.data import DataLoader
import json

from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os

import collections

from sklearn.neighbors import KNeighborsClassifier



parser = argparse.ArgumentParser("Center Loss Test")
# model
parser.add_argument('--model', type=str, default='cnn')
parser.add_argument('--point',type=int,default=500)
parser.add_argument('--result_num',type=int,default=0)
args = parser.parse_args()



#def create_feature_json(gallery_path,probe_path):
#    probe_conds = ['nm-05','nm-06','cl-01','cl-02','bg-01','bg-02']
#    gallery_conds = ['nm-01','nm-02','nm-03','nm-04']
#    angles = ['000','018','036','054','072','090','108','126','144','162','180']
#    for i in range(63,125):
#        idx = '%03d' % i
#        for angle in angles:
#            for cond in gallery_conds:
#                json_gallery = str(idx) + '-'+cond +'-' +angle+'.json'
#                save_gallery = gallery_path + json_gallery
#                print('save_gallery:',save_gallery)
##                os.mknod(save_gallery)
#                f  = open(save_gallery,'w')
#                f.close()
#                
#            for cond in probe_conds:
#                json_probe = str(idx) + '-'+cond +'-' +angle+'.json'
#                save_probe = probe_path + json_probe
#                print("save_probe:",save_probe)
##                os.mknod(save_probe)
#                f  = open(save_probe,'w')
#                f.close()
#                
#    
#    print('create end')
#                
            



def get_features(model,test_set,save_path):
    
#    gallery_dict = collections.OrderedDict()
    
#    gallery_conds = ['nm-01','nm-02','nm-03','nm-04']
#    angles = ['000','018','036','054','072','090',
#              '108','126','144','162','180']
    
#    for i in range(63,64):
#        for cond in gallery_conds:
#            for angle in angles:
#                gallery_key = str('%03d' % i) + '-' +cond + '-'+angle
#                gallery_dict[gallery_key]  = []
    
    if torch.cuda.is_available():
        model = model.cuda()
        print('===========model using cuda==========')
    else:
        print('===========model using cpu=============')
    
    model.eval()
    
    for img,label in test_set:
        if torch.cuda.is_available():
            img = Variable(img.cuda())
            label = np.array(label)
            print('============data using cuda==============')
        else:
            img = Variable(img)
            label = np.array(label)
            print('============data using cpu================')
            
        feature1,feature2,output = model(img)
        print('feature1.shape:{}'.format(feature1.shape))  #[batch_size,512]
        print('feature1:{}'.format(feature1))
        
        print('label.shape:{}'.format(label.shape))
        print('label.shape[0]:{}'.format(label.shape[0]))
        for i in range(label.shape[0]):
#            if(gallery_dict.has_key(label[i])):
            print('label:{}'.format(label[i]))
            feature_cpu  = feature1[i].cpu().detach().numpy()
            feature_cpu = feature_cpu.tolist()
#                gallery_dict[label[i]].append(feature_cpu)
                
            gallery_path = label[i] + '.json'
            gallery_data = {}
            gallery_data['feature'] = feature_cpu
            with open(os.path.join(save_path,gallery_path),'w') as w_file:
                json.dump(gallery_data,w_file)
                w_file.close()
        
    print('get feature end')



def knn_conds(gallery_set,probe_set):
    print('KNN Class Start')
    ix = 0
    result_nm = np.zeros((1,11))
    result_bg = np.zeros((1,11))
    result_cl = np.zeros((1,11))
    angles = ['000','018','036','054','072',
              '090',
              '108','126','144','162','180'
            ]
    gallery_conds = ['nm-01','nm-02','nm-03','nm-04']
    probe_nm_conds = ['nm-05','nm-06']
    probe_bg_conds = ['bg-01','bg-02']
    probe_cl_conds = ['cl-01','cl-02']
    
    for angle in angles:
        X = []
        y = []
        print('===========gallery====================')
        for p in range(63,125):
            for cond in gallery_conds:
                gallery_path = str('%03d' % p) + '-'+cond + '-'+angle+'.json'
                gallery_path = os.path.join(gallery_set,gallery_path)
                print('gallery_path:{}'.format(gallery_path))
                if os.path.exists(gallery_path):
                    gallery_feature =  json.load(open(gallery_path))['feature']
                    print('gallery_feature:{}'.format(gallery_feature))
                    X.append(gallery_feature)
                    y.append(p-63)
        
        nbrs = KNeighborsClassifier(n_neighbors=1,p=1,weights="distance")
        X = np.asarray(X)
        y = np.asarray(y).astype(np.int32)
        
        nbrs.fit(X,y)
        
        print('============probe   NM =====================')
        testX = []
        testy = []
        for p in range(63,125):
            for cond in probe_nm_conds:
                probe_path = str('%03d'% p) + '-'+cond + '-'+angle+'.json'
                probe_path = os.path.join(probe_set,probe_path)
                print('probe_path:{}'.format(probe_path))
                if os.path.exists(probe_path):
                    probe_feature = json.load(open(probe_path))['feature']
                    print("probe_feature:{}".format(probe_feature))
                    testX.append(probe_feature)
                    testy.append(p-63)
        testX = np.asarray(testX)
        testy = np.asarray(testy).astype(np.int32)
        s = nbrs.score(testX,testy)
        result_nm[0][ix] = s
        
        testX = []
        testy = []
        print('===========probe  BG=================')
        for p in range(63,125):
            for cond in probe_bg_conds:
               probe_path = str('%03d'% p) + '-'+cond + '-'+angle+'.json'
               probe_path = os.path.join(probe_set,probe_path)
               print('probe_path:{}'.format(probe_path))
               if os.path.exists(probe_path):
                   probe_feature = json.load(open(probe_path))['feature']
                   print("probe_feature:{}".format(probe_feature))
                   testX.append(probe_feature)
                   testy.append(p-63)
        testX = np.asarray(testX)
        testy = np.asarray(testy).astype(np.int32)
        s = nbrs.score(testX,testy)
        result_bg[0][ix] = s 
        

        testX = []
        testy = []
        print('==========probe CL=================')
        for p in range(63,125):
            for cond in probe_cl_conds:
               probe_path = str('%03d'% p) + '-'+cond + '-'+angle+'.json'
               probe_path = os.path.join(probe_set,probe_path)
               print('probe_path:{}'.format(probe_path))
               if os.path.exists(probe_path):
                   probe_feature = json.load(open(probe_path))['feature']
                   print("probe_feature:{}".format(probe_feature))
                   testX.append(probe_feature)
                   testy.append(p-63)
        testX = np.asarray(testX)
        testy = np.asarray(testy).astype(np.int32)
        s = nbrs.score(testX,testy)
        result_cl[0][ix] = s         
    
        ix += 1
    
#    print('result_nm:{}'.format(result_nm))
#    print('result_bg:{}'.format(result_bg))
#    print('result_cl:{}'.format(result_cl))
    result = np.concatenate((result_nm,result_bg),0)
    result = np.concatenate((result,result_cl),0)
    print('result.shape:{}'.format(result.shape))
    print('result:{}'.format(result))
    
    result_path = "/home/mg/code/GEI+PTSN/train/pytorch-center-loss-master/results/result_" + str(args.result_num)+'.csv'
    print('result_path:{}'.format(result_path))
    np.savetxt(result_path,result)
    
    
                    
        
                    
                    
                    
        
    
    


def main():
    
    #==========model loading==================
    model = models.create(name=args.model, num_classes=62)
    checkpoint = '/home/mg/code/GEI+PTSN/train/pytorch-center-loss-master/snapshots_512/snapshot_' + str(args.point) + '.t7'
    print("checkpoint:{}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model,device_ids = range(torch.cuda.device_count()-1))
        
        print("Total GPU numbers:",torch.cuda.device_count(),"Current device:",torch.cuda.current_device())
    
    
    model.load_state_dict(checkpoint['cnn'])
    
    #=============get gallery_set and probe_set features=================
    gallery_set = test_dset('/home/mg/code/data/GEI_B/test_rst/gallery/')
    gallery_data = DataLoader(gallery_set,batch_size=2,shuffle=False)
    get_features(model,gallery_data,'/home/mg/code/GEI+PTSN/train/pytorch-center-loss-master/features/gallery_features/')               
                       
    probe_set = test_dset('/home/mg/code/data/GEI_B/test_rst/probe/')
    probe_data = DataLoader(probe_set,batch_size = 2,shuffle = False)
    get_features(model,probe_data,'/home/mg/code/GEI+PTSN/train/pytorch-center-loss-master/features/probe_features/')  
    
    #============knn=======================
    knn_conds('/home/mg/code/GEI+PTSN/train/pytorch-center-loss-master/features/gallery_features/','/home/mg/code/GEI+PTSN/train/pytorch-center-loss-master/features/probe_features/')
    
    
    
if __name__ == '__main__':
    main()
    
        
        
        
        
        
    
        
    



