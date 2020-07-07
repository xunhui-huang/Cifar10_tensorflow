def base():
    from src.basic_cnn import Basic_CNN
    baseline=Basic_CNN()
    baseline.train(dataloader=baseline.cifar10, batch_size=baseline.batchsize, n_epoch=baseline.Epochs)

def spp():
    from src.SPPnet import SPPNET
    sppnet=SPPNET()
    sppnet.train(dataloader=sppnet.cifar10, batch_size=sppnet.batchsize, n_epoch=sppnet.Epochs)

def vgg():
    from src.vgg import VGG19
    vggnet=VGG19()
    vggnet.train()

def res():
    from src.resnet50 import Resnet50
    resnet=Resnet50()
    resnet.train()

if __name__=="__main__":
    res()

