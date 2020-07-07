"""亮点:增加了L2正则化"""
import tensorflow as tf
import time
import numpy as np
import os
import sys
#from src.data.cifar10 import Corpus
import argparse
import yaml

Root=os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(Root)

from data_preprocess.data_proc import data_proc
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Basic_CNN:
    def __init__(self):
        self.cifar10=data_proc()
        self.args=self.parse()
        self.batchsize=self.args.batch_size
        self.max_steps=self.args.iteration
        self.lr=self.args.learning_rate
        self.Epochs=self.args.n_epochs
        self.image_holder=tf.placeholder(shape=[None,24,24,3],dtype=tf.float32)                          
        self.label_holder=tf.placeholder(shape=[None],dtype=tf.int64)

    def parse(self):
        parse=argparse.ArgumentParser('Super Parameters')
        parse.add_argument('-b',dest='batch_size',default=128,required=False)
        parse.add_argument('-i',dest='iteration',default=10,required=False)
        parse.add_argument('-l',dest='learning_rate',default=0.001,required=False)
        parse.add_argument('-n',dest='n_epochs',default=2,required=False)
        args=parse.parse_args()
        return args

    def weight_init(self,shape,wl,name):
        '''对权重加上L2正则化，正则化大小有wl限定'''
        weight=tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=0.1,name=name))
        if weight is not None:
            weight_loss=tf.multiply(tf.nn.l2_loss(weight),wl,name="weight_loss")
            tf.add_to_collection('loss',weight_loss)
        return weight

    def bias_init(self,shape,name):
        return tf.Variable(initial_value=tf.constant(0.1,shape=shape,name=name))

    def conv2d(self,x,conv_w):
        return tf.nn.conv2d(x,conv_w,strides=[1,1,1,1],padding='SAME')

    def maxpool(self,x,ksize):
        return tf.nn.max_pool(x,[1,ksize,ksize,1],strides=[1,ksize,ksize,1],padding='SAME')

    def LRN(self,x):
        return tf.nn.lrn(x)

    def inference(self,x):
        with open("config/basic_cnn.yaml","r") as fo:
            network_options=yaml.load(fo)
        ##conv1
        conv_1_dict=network_options["net"]["conv"][0]
        pool_1_dict=network_options["net"]["conv"][1]
        weights_conv1=self.weight_init([conv_1_dict["x_size"],conv_1_dict["y_size"],3,conv_1_dict["n_filters"]],0.0,'conv1_w')
        bias_conv1=self.bias_init([conv_1_dict["n_filters"]],'conv1_b')
        conv1_out=tf.nn.relu(self.conv2d(x,weights_conv1)+bias_conv1)
        pool1_out=self.maxpool(conv1_out,pool_1_dict["x_stride"])
        ##conv2
        conv_2_dict=network_options["net"]["conv"][2]
        pool_2_dict=network_options["net"]["conv"][3]
        weights_conv2=self.weight_init([conv_2_dict["x_size"],conv_2_dict["y_size"],conv_1_dict["n_filters"],conv_2_dict["n_filters"]],0.05,'conv2_w')
        bias_conv2=self.bias_init([conv_2_dict["n_filters"]],'conv2_b')
        conv2_out=tf.nn.relu(self.conv2d(pool1_out,weights_conv2)+bias_conv2)
        pool2_out=self.maxpool(conv2_out,pool_2_dict["x_stride"])

        #flatten=tf.reshape(pool2_out,[batchsize,-1])
        ####fc1
        fc_1_dict=network_options["net"]["fc"][0]
        flatten=tf.reshape(pool2_out,[-1,6*6*64])
        dim=flatten.get_shape()[1].value
        weights_fc1=self.weight_init([dim,fc_1_dict["hidden_dim"]],0.004,'fc1_w')
        bias_fc1=self.bias_init([fc_1_dict["hidden_dim"]],'fc1_b')
        fc1_out=tf.nn.relu(tf.matmul(flatten,weights_fc1)+bias_fc1)
        ###fc2
        fc_2_dict=network_options["net"]["fc"][1]
        weights_fc2=self.weight_init([fc_1_dict["hidden_dim"],fc_2_dict["hidden_dim"]],0.0,'fc2_w')
        bias_fc2=self.bias_init([fc_2_dict["hidden_dim"]],'fc2_b')
        #output=tf.nn.softmax(tf.matmul(fc1_out,weights_fc2)+bias_fc2)
        output=tf.matmul(fc1_out,weights_fc2)+bias_fc2
        return output

    def loss_l2(self,logits,labels):
        #labels=tf.cast(labels,tf.int64)
        #cross_entropy=tf.reduce_mean(-tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,0.0001,1))))
        cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels),name='cross_entropy')
        tf.add_to_collection('loss',cross_entropy)
        return tf.add_n(tf.get_collection('loss'),name='total_loss')

    def train(self,dataloader, n_epoch=None,batch_size=None):
        logits=self.inference(self.image_holder)
        loss=self.loss_l2(logits,self.label_holder)
        train_op=tf.train.AdamOptimizer(self.lr).minimize(loss)
        top_k_acc=tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits,self.label_holder,5),tf.float32))

        init_op=tf.global_variables_initializer()
        local_init=tf.local_variables_initializer()

        gpu_options=tf.GPUOptions(allow_growth=True)
        config=tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            sess.run(local_init)
            #模型训练
            for epoch in range(n_epoch):

                # 数据增强
                train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
                    flip=True, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
                train_labels = dataloader.train_labels
                valid_images = dataloader.data_augmentation(dataloader.valid_images, mode='valid',
                    flip=True, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
                valid_labels = dataloader.valid_labels
                test_images=dataloader.data_augmentation(dataloader.test_images,mode='test',crop=True,crop_shape=(24,24,3))
                test_labels=dataloader.test_labels
                
                for i in range(self.max_steps):
                    batch_images = train_images[i: i+batch_size]
                    batch_labels = train_labels[i: i+batch_size]
                    batch_loss,_=sess.run([loss,train_op],feed_dict={self.image_holder:batch_images,self.label_holder:batch_labels})
                    print("Iteration %d,train loss %.4f" % (i,batch_loss))
                
                batch_images_val=valid_images[epoch:epoch+batch_size]
                batch_labels_val=valid_labels[epoch:epoch+batch_size]
                batch_acc_val,batch_loss_val=sess.run([top_k_acc,loss],feed_dict={self.image_holder:batch_images_val,self.label_holder:batch_labels_val})
                #print('Epoch:%d,batch_acc is %.4f' % (i,batch_acc))
                print('Epoch:',epoch)
                print('accuracy in valid dataset is:',batch_acc_val)
                #print('Epoch:%d,batch_loss is %.4f' % (i,batch_loss_te))
                print('loss in valid dataset is:',batch_loss_val)
                test_image=test_images[epoch:epoch+2]
                #test_image=np.reshape(test_images[epoch],(1,test_images[epoch].shape[0],test_images[epoch].shape[1],test_images[epoch].shape[2]))
                #print(test_image.shape)
                test_label=test_labels[epoch:epoch+2]
                acc_test_per,loss_test_per=sess.run([top_k_acc,loss],feed_dict={self.image_holder:test_image,self.label_holder:test_label})
                print('accuracy in test dataset is:',acc_test_per)
                print('loss in test dataset is:',loss_test_per)

