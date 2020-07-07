import tensorflow as tf
import time
import numpy as np
import os
import sys
import argparse
import yaml
import math
from tensorflow.python.framework import graph_util

if True:
    from data_preprocess.data_proc import data_proc
    from tf_layers.ConvLayer import ConvLayer
    from tf_layers.PoolLayer import PoolLayer
    from tf_layers.FcLayer import FcLayer

model_path="models/"
if not os.path.exists(model_path):
    os.makedirs(model_path)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Resnet50:
    def __init__(self):
        with open("config/resnet50.yaml","r") as fo:
            self.network_options=yaml.load(fo)
        self.args=self.parse()
        self.batch_size=self.args.batchsize
        self.lr=self.args.learning_rate
        self.it=self.args.iteration
        self.epochs=self.args.Epochs
        self.discard_prob=tf.placeholder(tf.float32,name="dropout_discard_prob")
        self.inputs=tf.placeholder(dtype=tf.float32,shape=[None,24,24,3],name="input_image")
        self.labels=tf.placeholder(dtype=tf.int64,shape=[None],name="labels")
        self.cifar10=data_proc()
        self.networks()

    def parse(self):
        parse=argparse.ArgumentParser("Super Parameters")
        parse.add_argument("-b",dest="batchsize",default=128,required=False)
        parse.add_argument("-l",dest="learning_rate",default=1e-4,required=False)
        parse.add_argument("-i",dest="iteration",default=5,required=False)
        parse.add_argument("-n",dest="Epochs",default=2,required=False)
        args=parse.parse_args()
        return args
    
    def networks(self):
        self.conv1_list=[]
        self.pool1_list=[]
        self.conv2_list=[]
        self.conv3_list=[]
        self.conv4_list=[]
        self.conv5_list=[]
        self.residual_block=[]
        self.residual_block.append(self.conv2_list)
        self.residual_block.append(self.conv3_list)
        self.residual_block.append(self.conv4_list)
        self.residual_block.append(self.conv5_list)       
        self.denselist=[]
        for conv_first in self.network_options["net"]["conv_first"]:
            layer=ConvLayer(filters=conv_first["n_filters"],kernel_size=[conv_first["x_size"],conv_first["y_size"]],
            strides=[conv_first["x_stride"],conv_first["y_stride"]],activation=conv_first["activation"],
            name=conv_first["name"],bn=conv_first["bn"])
            self.conv1_list.append(layer)
        
        for pool_first in self.network_options["net"]["pool_first"]:
            layer=PoolLayer(kernel_size=[pool_first["x_size"],pool_first["y_size"]],strides=[pool_first["x_stride"],pool_first["y_stride"]],
            mode=pool_first["mode"],name=pool_first["name"])
            self.pool1_list.append(layer)
        
        for index in range(self.network_options["residual_block_num"]):
            for conv_x in self.network_options["net"][self.network_options["residual_block"][index]]:
                layer=ConvLayer(filters=conv_x["n_filters"],kernel_size=[conv_x["x_size"],conv_x["y_size"]],
                strides=[conv_x["x_stride"],conv_x["y_stride"]],activation=conv_x["activation"],
                name=conv_x["name"],bn=conv_x["bn"])
                self.residual_block[index].append(layer)

        for dense in self.network_options["net"]["dense_first"]:
            layer=FcLayer(hidden_dim=dense["hidden_dim"],activation=dense["activation"],name=dense["name"],
            dropout=dense["dropout"],bn=dense["bn"],discard_prob=self.discard_prob) 
            self.denselist.append(layer)
        

    def inference(self,inputs):
        for conv1_layer in self.conv1_list:
            self.conv1_map=conv1_layer.get_output(inputs=inputs)
        for pool1_layer in self.pool1_list:
            self.pool1_map=pool1_layer.get_output(inputs=self.conv1_map)
        self.conv_x_map=self.pool1_map
        for block_num in range(self.network_options["residual_block_num"]):
            for conv_num in range(self.network_options["conv_x_num"][block_num]):
                if conv_num<self.network_options["conv_x_num"][block_num]-1:
                    self.conv_x_1_map=self.residual_block[block_num][0].get_output(inputs=self.conv_x_map)
                    self.conv_x_2_map=self.residual_block[block_num][1].get_output(inputs=self.conv_x_1_map)
                    self.conv_x_3_map=self.residual_block[block_num][2].get_output(inputs=self.conv_x_2_map)
                    self.conv_x_map=tf.nn.relu(self.conv_x_3_map+self.conv_x_map)
                else:
                    self.conv_x_1_map=self.residual_block[block_num][3].get_output(inputs=self.conv_x_map)
                    self.conv_x_2_map=self.residual_block[block_num][4].get_output(inputs=self.conv_x_1_map)
                    self.conv_x_3_map=self.residual_block[block_num][5].get_output(inputs=self.conv_x_2_map)
                    self.conv_x_block_map=self.residual_block[block_num][6].get_output(inputs=self.conv_x_map)
                    self.conv_x_map=tf.nn.relu(self.conv_x_3_map+self.conv_x_block_map)
        self.conv_size=self.conv_x_map.get_shape()[1].value
        self.flatten=tf.reshape(self.conv_x_map,[-1,self.conv_size*self.conv_size*256])
        for fc_layer in self.denselist:
            self.fc1=fc_layer.get_output(inputs=self.flatten)
        output=tf.identity(self.fc1,name="output")

        return output
    
    def loss(self,output):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=self.labels),name="cross_entropy")
    
    def accuracy(self,output,n):
        return tf.reduce_mean(tf.cast(tf.nn.in_top_k(output,self.labels,n),tf.float32),name="accuracy")
    
    def train(self):
        self.output=self.inference(self.inputs)
        train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss(self.output))
        loss=self.loss(self.output)
        acc_5=self.accuracy(self.output,5)
        init_op=tf.global_variables_initializer()
        local_init_op=tf.local_variables_initializer()

        self.gpu_options=tf.GPUOptions(allow_growth=True)
        self.config=tf.ConfigProto(gpu_options=self.gpu_options)

        with tf.Session(graph=tf.get_default_graph(),config=self.config) as sess:
            sess.run(init_op)
            sess.run(local_init_op)

            for epoch in range(self.epochs):
                #数据增强
                self.train_images=self.cifar10.data_augmentation(self.cifar10.train_images,flip=True,mode="train",
                                                                crop=True,crop_shape=(24,24,3),whiten=True,noise=False)
                self.train_labels=self.cifar10.train_labels
                self.valid_images=self.cifar10.data_augmentation(self.cifar10.valid_images,mode="valid",
                                                                crop=True,crop_shape=(24,24,3))
                self.valid_labels=self.cifar10.valid_labels
                self.test_images=self.cifar10.data_augmentation(self.cifar10.test_images,mode="test",
                                                                crop=True,crop_shape=(24,24,3))
                self.test_labels=self.cifar10.test_labels

                #迭代训练
                for i in range(self.it):
                    self.batch_images=self.train_images[i:i+self.batch_size]
                    self.batch_labels=self.train_labels[i:i+self.batch_size]
                    train_loss,_=sess.run([loss,train_op],feed_dict={self.inputs:self.batch_images,self.labels:self.batch_labels})
                    print("Iteration %d ,train loss is %f" % (i,train_loss))
                self.batch_val_images=self.valid_images[epoch*50:epoch*50+self.batch_size]
                self.batch_val_labels=self.valid_labels[epoch*50:epoch*50+self.batch_size]
                valid_loss,valid_acc=sess.run([loss,acc_5],feed_dict={self.inputs:self.batch_val_images,self.labels:self.batch_val_labels})
                print("Epoch %d ,valid loss is %f" % (epoch,valid_loss))
                print("Epoch %d ,valid acc is %f" % (epoch,valid_acc)) 





                    

                    
                
