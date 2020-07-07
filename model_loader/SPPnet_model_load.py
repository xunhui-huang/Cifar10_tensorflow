import tensorflow as tf
from tensorflow.python.framework import graph_util
from mycode.data_preprocess.data_proc import data_proc

cifar10=data_proc()
test_images=cifar10.data_augmentation(cifar10.test_images,mode='test',crop=True,crop_shape=(24,24,3))
test_images=test_images[0:2]
tf.reset_default_graph()        #重置计算图

model_path="mycode/models/spp_model.pb"
def network_reload(model_path,test_images):
    with tf.Session() as sess:
        output_graph_def=tf.GraphDef()
        #获得默认图
        graph=tf.get_default_graph()
        with open(model_path,"rb") as fo:
            output_graph_def.ParseFromString(fo.read())
            _=tf.import_graph_def(output_graph_def,name="") #导入计算图
            #print("%d op nodes in current graph " % len(output_graph_def.node))
            summarywriter=tf.summary.FileWriter("log_dir")
            summarywriter.add_graph(graph)
        sess.run(tf.global_variables_initializer())        #全局变量初始化
        
        tensor_image=graph.get_tensor_by_name("images:0")
        tensor_output=graph.get_tensor_by_name("output:0")

        output=sess.run(tensor_output,feed_dict={tensor_image:test_images})
        print(output)

if __name__=='__main__':
    network_reload(model_path,test_images)
        