import tensorflow as tf

class ConvLayer:
    def __init__(self,filters,kernel_size,strides,activation,name,bn):
        self.filters=filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.name=name
        self.activation=activation
        self.bn=bn
        self.padding="same"
        self.use_bias=True
        self.conv2d=tf.layers.Conv2D(filters=self.filters,kernel_size=self.kernel_size,strides=self.strides,
                                        padding=self.padding,activation=None,name=self.name,use_bias=self.use_bias)
        self.batch_normal=tf.layers.BatchNormalization(axis=-1,momentum=0.9,epsilon=1e-5,center=True,
                                                            scale=True,trainable=True,name='%s_bn'%(self.name))
    def get_output(self,inputs):
        self.hidden=self.conv2d(inputs=inputs)
        if self.bn:
            self.hidden=self.batch_normal(inputs=self.hidden)
            
        ##激活
        if self.activation=="sigmoid":
            output=tf.nn.sigmoid(self.hidden)
        elif self.activation=="relu":
            output=tf.nn.relu(self.hidden)
        elif self.activation=="tanh":
            output=tf.nn.tanh(self.hidden)
        elif self.activation=="leaky_relu":
            output=tf.nn.leaky_relu(self.hidden)
        return output

