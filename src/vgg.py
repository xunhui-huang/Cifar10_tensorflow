"""亮点:增加了L2正则化"""
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

class VGG19:
    def __init__(self):
        with open("config/vgg.yaml","r") as fo:
            self.network_options=yaml.load(fo)
        self.args=self.parse()
        self.batchsize=self.args.batch_size
        self.max_steps=self.args.iteration
        self.lr=self.args.learning_rate
        self.Epochs=self.args.n_epochs
        self.discard_prob=tf.placeholder(tf.float32,name="dropout_discard_prob")
        self.inputs=tf.placeholder(shape=[None,24,24,3],dtype=tf.float32,name="inputs")
        self.labels=tf.placeholder(shape=[None],dtype=tf.int64,name="labels")
        self.network()
        self.output=self.inference(input=self.inputs)
        self.cifar10=data_proc()

    def parse(self):
        parse=argparse.ArgumentParser('Super Parameters')
        parse.add_argument('-b',dest='batch_size',default=128,required=False)
        parse.add_argument('-i',dest='iteration',default=10,required=False)
        parse.add_argument('-l',dest='learning_rate',default=0.001,required=False)
        parse.add_argument('-n',dest='n_epochs',default=2,required=False)
        args=parse.parse_args()
        return args

    def network(self):
        self.Convlist=[]
        self.Fclist=[]
        for conv_dict in self.network_options["net"]["conv"]:
            if conv_dict["type"]=="conv":
                layer=ConvLayer(filters=conv_dict["n_filter"],kernel_size=[conv_dict["x_size"],conv_dict["y_size"]],
                                strides=[conv_dict["x_stride"],conv_dict["y_stride"]],activation=conv_dict["activation"],
                                name=conv_dict["name"],bn=conv_dict["bn"])
            elif conv_dict["type"]=="pool":
                layer=PoolLayer(kernel_size=[conv_dict["x_size"],conv_dict["y_size"]],strides=[conv_dict["x_stride"],conv_dict["y_stride"]],
                                mode=conv_dict["mode"],name=conv_dict["name"])
            self.Convlist.append(layer)
        for fc_dict in self.network_options["net"]["fc"]:
            layer=FcLayer(hidden_dim=fc_dict["hidden_dim"],activation=fc_dict["activation"],name=fc_dict["name"],
                            dropout=fc_dict["dropout"],bn=fc_dict["bn"],discard_prob=self.discard_prob)
            self.Fclist.append(layer)

    def inference(self,input):
        self.hidden=input
        for conv_layer in self.Convlist:
            self.hidden=conv_layer.get_output(inputs=self.hidden)
        self.hidden_size=self.hidden.get_shape()[1].value
        self.hidden=tf.reshape(self.hidden,[-1,self.hidden_size*self.hidden_size*256])
        for fc_layer in self.Fclist:
            self.hidden=fc_layer.get_output(inputs=self.hidden)
        self.out=tf.identity(self.hidden,name="output")
        return self.out

    def loss(self):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output,labels=self.labels),name="cross_entropy")
    
    def accuracy(self,top_n):
        return tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.output,self.labels,top_n),tf.float32))
    
    def train(self):
        self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss())
        self.init_op=tf.global_variables_initializer()
        self.local_init=tf.local_variables_initializer()
        
        self.gpu_options=tf.GPUOptions(allow_growth=True)
        self.config=tf.ConfigProto(gpu_options=self.gpu_options)
        with tf.Session(graph=tf.get_default_graph(),config=self.config) as sess:
            sess.run(self.init_op)
            sess.run(self.local_init)
            print([n.name for n in sess.graph.as_graph_def().node])
            #constant_graph=graph_util.convert_variables_to_constants(sess,sess.graph_def,"output")
            #模型训练
            for epoch in range(self.Epochs):
                #数据增强
                self.train_images=self.cifar10.data_augmentation(self.cifar10.train_images,mode="train",flip=True,
                                                                    crop=True,crop_shape=(24,24,3),whiten=True,noise=False)
                self.train_labels=self.cifar10.train_labels
                self.valid_images=self.cifar10.data_augmentation(self.cifar10.valid_images,mode="valid",flip=True,
                                                                    crop=True,crop_shape=(24,24,3),whiten=True,noise=False)
                self.valid_labels=self.cifar10.valid_labels
                self.test_images=self.cifar10.data_augmentation(self.cifar10.test_images,mode="test",crop=True,crop_shape=(24,24,3))
                self.test_labels=self.cifar10.test_labels

                for i in range(self.max_steps):
                    self.batch_images=self.train_images[i:i+self.batchsize]
                    self.batch_labels=self.train_labels[i:i+self.batchsize]
                    batch_top_n_acc,batch_loss=sess.run([self.accuracy(top_n=1),self.train_op],feed_dict={self.inputs:self.batch_images,self.labels:self.batch_labels,self.discard_prob:0.5})
                self.batch_images_val=self.valid_images[epoch:epoch+self.batchsize]
                self.batch_labels_val=self.valid_labels[epoch:epoch+self.batchsize]
                batch_top_n_acc_val,batch_loss_val=sess.run([self.accuracy(top_n=1),self.loss()],feed_dict={self.inputs:self.batch_images_val,self.labels:self.batch_labels_val,self.discard_prob:0.0})
                print('Epoch:%d,batch_loss is %.4f' % (epoch,batch_loss_val))
                print('Epoch:%d,accuracy is %.4f' % (epoch,batch_top_n_acc_val))
            #with tf.gfile.FastGFile(model_path+"vgg19_model.pb","wb") as fo:
                #fo.write(constant_graph.SerializeToString())
        

