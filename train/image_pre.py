# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:55:19 2019

@author: Administrator
"""
import os
import shutil

"""
copy_file:
    将一个文件夹下所有文件夹包含的文件复制到一个文件夹中
"""
def copy_file(src_dir,dst_dir):
    for f in os.listdir(src_dir):
        src_f = os.path.join(src_dir,f)


        if os.path.isfile(src_f):
            dst_f = os.path.join(dst_dir,f)            
            shutil.copy(src_f,dst_f)
        
        if os.path.isdir(src_f):
            copy_file(src_f,dst_dir)
    
    print('copy end')


"""
   all_copy:
       将文件夹下的所有文件夹中的内容按照copy_file处理
"""

def all_copy(src_root,dst_root):
    for i in range(1,3):
        idx = '%03d' % i 
        src_path = src_root + idx
        dst_path = dst_root + idx
        
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        copy_file(src_path,dst_path)
    print('all_copy  end')          


if __name__ == '__main__':
#    src_path = 'D:\\@mmg\\GEI_CASIA_B\\gei\\001\\'
#    dst_path = 'E:\\CASIA_B_GEI\\001\\'
#    copy_file(src_path,dst_path)
    
    all_copy('D:\\@mmg\\GEI_CASIA_B\\gei\\','E:\\CASIA_B_GEI\\')
        
