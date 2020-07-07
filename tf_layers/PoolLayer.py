import tensorflow as tf

class PoolLayer:
    def __init__(self,kernel_size,strides,mode,name):
        self.kernel_size=kernel_size
        self.strides=strides
        self.mode=mode
        self.name=name
        if self.mode=="max":
            self.pool=tf.layers.MaxPooling2D(pool_size=self.kernel_size,strides=self.strides,padding="same",name=self.name)
        elif self.mode=="avg":
            self.pool=tf.layers.AveragePooling2D(pool_size=self.kernel_size,strides=self.strides,padding="same",name=self.name)
    def get_output(self,inputs):
        self.output=self.pool(inputs=inputs)
        return self.output
