import pickle
import os
import platform
import numpy as np
from cv2 import cv2
import random

class data_proc:
    def __init__(self):
        self.file_dir='src/data/CIFAR10_data'
        self.data_loader()
        self.train_and_valid_split(prob=0.9)
    def data_loader(self):
        '''读取数据集'''
        images,labels=[],[]
        #读取训练集
        train_files=['%s/data_batch_%d' % (self.file_dir,j) for j in range(1,2)]
        for filename in train_files:
            with open(filename,'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10=pickle.load(fo,encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10=pickle.load(fo)
            for i in range(len(cifar10[b'labels'])):
                image=cifar10[b'data'][i]
                image=np.reshape(image,[3,32,32])
                image=np.transpose(image,(1,2,0))
                image=image.astype(float)
                images.append(image)
            labels.extend(cifar10[b'labels'])
        self.train_images=np.array(images,dtype='float')
        self.train_labels=np.array(labels,dtype='int')
        
        #读取测试集
        images,labels=[],[]
        test_file=self.file_dir+'/test_batch'
        with open(test_file,'rb') as fo:
            if 'Window' in platform.platform():
                cifar10=pickle.load(fo,encoding='bytes')
            elif 'Linux' in platform.platform():
                cifar10=pickle.load(fo)
        for i in range(len(cifar10[b'labels'])):
            image=cifar10[b'data'][i]
            image=np.reshape(image,(3,32,32))
            image=np.transpose(image,(1,2,0))
            image=image.astype(float)
            images.append(image)
        labels.extend(cifar10[b'labels'])
        self.test_images=np.array(images,dtype='float')
        self.test_labels=np.array(labels,dtype='int')
    def train_and_valid_split(self,prob=0.9):
        """将训练集划分为训练集和验证集"""
        images,labels=self.train_images,self.train_labels
        thresh=int(prob*labels.shape[0])
        self.train_images=images[0:thresh,:,:,:]
        self.train_labels=labels[0:thresh]
        self.valid_images=images[thresh:,:,:,:]
        self.valid_labels=labels[thresh:]
    def data_augmentation(self,images,mode='train',flip=False,crop=False,crop_shape=(24,24,3),
                            whiten=False,noise=False,noise_mean=0.0,noise_std=0.01):
        if crop:
            if mode=='train':
                images=self.crop(images,crop_shape)
            elif mode=='valid':
                images=self.crop(images,crop_shape)
            elif mode=='test':
                images=self.crop(images,crop_shape)
        if flip:
            images=self.flip(images)
        if whiten:
            images=self.whiten(images)
        if noise:
            images=self.noise(images,noise_mean,noise_std)
        return images
    def crop(self,images,crop_size):
        #图像裁剪
        new_images=[]
        for i in range(images.shape[0]):
            image=images[i,:,:,:]
            left=random.randint(0,7)
            top=random.randint(0,7)
            crop_image=image[left:left+crop_size[0],top:top+crop_size[1],:]
            new_images.append(crop_image)
        return np.array(new_images)
    
    def flip(self,images):
        #图片翻转
        for i in range(images.shape[0]):
            old_image=images[i,:,:,:]
            if random.random()<0.5:
                new_image=cv2.flip(old_image,1)
            else:
                new_image=old_image
            images[i,:,:,:]=new_image
        
        return images
    def whiten(self,images):
        #图片白化
        for i in range(images.shape[0]):
            old_image=images[i,:,:,:]
            new_image=(old_image-np.mean(old_image))/np.std(old_image)
            images[i,:,:,:]=new_image
        return images
    def noise(self,images,noise_mean,noise_std):
        #图片加高斯噪声
        for i in range(images.shape[0]):
            old_image=images[i,:,:,:]
            for j in range(old_image.shape[0]):
                for k in range(old_image.shape[1]):
                    for n in range(old_image.shape[2]):
                        new_image=old_image[j,k,n]+random.gauss(noise_mean,noise_std)
            images[i,:,:,:]=new_image
        return images
            
