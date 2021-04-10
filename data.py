import numpy as np

#import cv2

from keras.datasets import mnist,fashion_mnist,cifar10,cifar100

class Dataset(object):

    def __init__(self ):

        self.imageShape = None

        self.data = {}

    def ImageShape(self):

        return  self.imageShape

    def ConvertToCifarFormat(self,data):

        dataList = []
        for i,img in enumerate(data):

            img32 = cv2.resize(img,(32,32))
            imgRGB = cv2.cvtColor(img32,cv2.COLOR_GRAY2RGB)
            dataList.append(imgRGB)

        return  np.array(dataList)

    def InitializeData(self):

        # normalize the images
        # reshape labels to 1D
        # convert mnist and fashion to 32x32x3

        # (_,_), (test_images, test_labels) = mnist.load_data()
        # self.data["mnist"] = {"test_images":self.ConvertToCifarFormat(test_images)/ 255.0,
        #                       "test_labels":test_labels.reshape(-1)}
        #
        # (_,_), (test_images, test_labels) = fashion_mnist.load_data()
        # self.data["fashion_mnist"] = {"test_images":self.ConvertToCifarFormat(test_images)/ 255.0,
        #                               "test_labels":test_labels.reshape(-1)}

        (_,_), (test_images, test_labels) = cifar10.load_data()
        self.data["cifar10"] = {"test_images":test_images/ 255.0,
                                "test_labels":test_labels.reshape(-1)}

        # (_,_), (test_images, test_labels) = cifar100.load_data()
        # self.data["cifar100"] = {"test_images":test_images/ 255.0,
        #                          "test_labels":test_labels.reshape(-1)}



        self.imageShape = [32,32,3]

    def TestData(self,id):
        return self.data[id]["test_images"], self.data[id]["test_labels"]





