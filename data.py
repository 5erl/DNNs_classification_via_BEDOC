
import copy
import os,sys
import bincode
import numpy as np
import random
import math
import cv2
import time


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




    def create_labels_loss(self,spreadCode,labels,labelsStartState,useOriginal=False):


        if spreadCode.BinCode().GetDataType() == "euclide_softmax":
            return  self.create_labels_loss_one_hot_select_multi(labels,spreadCode)

        elif self.lossType == 26 or self.lossType == 27 or self.lossType == 30 :

            return self.create_labels_loss26(spreadCode,labels,labelsStartState,useOriginal)

        elif self.lossType == 28:
            if spreadCode.BinCode().GetDataType() != "one_hot_multi_all":
                return  self.create_labels_loss_one_hot_select_multi(labels,spreadCode)
            else:
                return  self.create_labels_loss_one_hot_select_multi_all(labels,spreadCode)

        elif self.lossType == 9 or self.lossType == 4 or self.lossType == 2 or self.lossType == 7 or self.lossType == 1:
            if spreadCode.BinCode().GetDataType() != "one_hot_select":
                return  self.create_labels_loss9(labels,spreadCode)
            else:
                return  self.create_labels_loss_one_hot_select(labels,spreadCode)

        else:
            print("ERROR: create_labels_loss unknown loss type: %s"%self.lossType)
            exit(1)


    def initialize_auto_annotations(self):


        num_splits = len(self.data["train_split_indexes"])


        if "data_format" in self.data and self.data['data_format'] == 2:

                indexes = self.data["train_split_indexes"][0]

                self.data["actual_train_indexes"] = indexes

                self.data["actual_train_images"] = self.data["train_images"][indexes]
                self.data["actual_train_codes_num"] = self.data["train_codes_num"][indexes]
                self.data["actual_train_codes_loss"] = self.data["train_codes_loss"][indexes]
                self.data["actual_train_codes_annot_num"] = self.data["train_codes_num"][indexes]


                self.data["actual_test_images"] = self.data["valid_images"]
                self.data["actual_test_codes_num"] = self.data["valid_labels"]
                self.data["actual_test_codes_loss"] = None


                indexes = self.data["train_split_indexes"][1]
                self.data["actual_annot_indexes"] = indexes
                self.data["actual_annot_images"] = self.data["train_images"][indexes]
                self.data["actual_annot_codes_num"] = self.data["train_codes_num"][indexes]
                self.data["actual_annot_codes_loss"] = self.data["train_codes_loss"][indexes]
                self.data["actual_annot_codes_annot_num"] = self.data["train_codes_num"][indexes]



                self.data['valid_images'] = self.data['valid_images_full'] / 255.0
                self.data['valid_codes_num'] = self.data["valid_labels_full"]






        else:
            print("DG: !WARNING! NUM SPLITS= %d --->  ONLY TRAIN DATA!!!"%num_splits)


            indexes = self.data["train_split_indexes"][0]

            self.data["actual_train_indexes"] = indexes

            self.data["actual_train_images"] = self.data["train_images"][indexes]
            self.data["actual_train_codes_num"] = self.data["train_codes_num"][indexes]
            self.data["actual_train_codes_loss"] = self.data["train_codes_loss"][indexes]
            self.data["actual_train_codes_annot_num"] = self.data["train_codes_num"][indexes]

            self.data["actual_test_indexes"] = None
            self.data["actual_test_images"] = None
            self.data["actual_test_codes_num"] = None
            self.data["actual_test_codes_loss"] = None

            self.data["actual_annot_indexes"] = None
            self.data["actual_annot_images"] = None
            self.data["actual_annot_codes_num"] = None
            self.data["actual_annot_codes_loss"] = None
            self.data["actual_annot_codes_annot_num"] = None









