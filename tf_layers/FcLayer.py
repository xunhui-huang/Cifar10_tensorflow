import tensorflow as tf

class FcLayer:
    def __init__(self,hidden_dim,activation,name,dropout,bn,discard_prob):
        self.hidden_dim=hidden_dim
        self.activation=activation
        self.name=name
        self.dropout=dropout
        self.bn=bn
        self.discard_prob=discard_prob
        self.fc=tf.layers.Dense(units=self.hidden_dim,activation=None,name=self.name)
        self.batch_normal=tf.layers.BatchNormalization(axis=-1,momentum=0.9,epsilon=1e-5,name="%s_bn"%(self.name))
    def get_output(self,inputs):
        self.hidden=self.fc(inputs=inputs)
        if self.bn:
            self.hidden=self.batch_normal(inputs=self.hidden)
        #Dropout层
        if self.dropout:
            self.hidden=tf.layers.dropout(inputs=self.hidden,rate=self.discard_prob)      #参数 rate ：每一个元素丢弃的概率
        #激活函数
        if self.activation=="sigmoid":
            self.output=tf.nn.sigmoid(self.hidden)
        elif self.activation=="relu":
            self.output=tf.nn.relu(self.hidden)
        elif self.activation=="tanh":
            self.output=tf.nn.tanh(self.hidden)
        elif self.activation=="leaky_relu":
            self.output=tf.nn.leaky_relu(self.hidden)
        elif self.activation=="None":
            self.output=self.hidden
        return self.output
