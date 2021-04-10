#!/usr/bin/python3

import networks
import bincode
import data
import os


def main():


    parameters = [

        #SOFTMAX ONEHOT
        {"network_id": "cnn1_onehot",
         "network_activation":"softmax",
         "network_weights":"cnn1_onehot_softmax",
         "code_id":"one_hot",
         "data_type":"one_hot",
         "data_type_fca":"one_hot_all",
         "code_width":10,
         "name":"CNN1 - ONE_HOT - SOFTMAX",
         "index":3 },


        {"network_id": "cnn2_onehot",
         "network_activation":"softmax",
         "network_weights":"cnn2_onehot_softmax",
         "code_id":"one_hot",
         "data_type":"one_hot",
         "data_type_fca":"one_hot_all",
         "code_width":10,
         "name":"CNN2 - ONE_HOT - SOFTMAX",
         "index":9 },

        {"network_id": "ResNet20v2_onehot",
         "network_activation":"softmax",
         "network_weights":"resnet20v2_onehot_softmax",
         "code_id":"one_hot",
         "data_type":"one_hot",
         "data_type_fca":"one_hot_all",
         "code_width":10,
         "name":"ResNet20v2 - ONE_HOT - SOFTMAX",
         "index":13 },

        #ZADEH ONEHOT

        {"network_id": "cnn1_zadeh",
         "network_activation":"sigmoid",
         "network_weights":"cnn1_onehot_zadeh",
         "code_id":"one_hot",
         "data_type":"one_hot_zadeh",
         "data_type_fca":"one_hot_zadeh_all",
         "code_width":10,
         "name":"CNN1 - ONE_HOT - ZADEH",
         "index":2 },

        {"network_id": "cnn2_zadeh",
         "network_activation":"sigmoid",
         "network_weights":"cnn2_onehot_zadeh",
         "code_id":"one_hot",
         "data_type":"one_hot_zadeh",
         "data_type_fca":"one_hot_zadeh_all",
         "code_width":10,
         "name":"CNN2 - ONE_HOT - ZADEH",
         "index":8 },

        {"network_id": "ResNet20v2_zadeh",
         "network_activation":"one_zero_norm",
         "network_weights":"resnet20v2_onehot_zadeh",
         "code_id":"one_hot",
         "data_type":"one_hot_zadeh",
         "data_type_fca":"one_hot_zadeh_all",
         "code_width":10,
         "name":"ResNet20v2 - ONE_HOT - ZADEH",
         "index":12 },

        #ZADEH CRC7

        {"network_id": "cnn1_zadeh",
         "network_activation":"one_zero_norm",
         "network_weights":"cnn1_crc7_zadeh",
         "code_id":"crc7",
         "data_type":"one_hot_zadeh",
         "data_type_fca":"one_hot_zadeh_all",
         "code_width":10,
         "name":"CNN1 - CRC7 - ZADEH",
         "index":1 },

        {"network_id": "cnn2_zadeh",
         "network_activation":"one_zero_norm",
         "network_weights":"cnn2_crc7_zadeh",
         "code_id":"crc7",
         "data_type":"one_hot_zadeh",
         "data_type_fca":"one_hot_zadeh_all",
         "code_width":10,
         "name":"CNN2 - CRC7 - ZADEH",
         "index":7 },

        {"network_id": "ResNet20v2_zadeh",
         "network_activation":"one_zero_norm",
         "network_weights":"resnet20v2_crc7_zadeh",
         "code_id":"crc7",
         "data_type":"one_hot_zadeh",
         "data_type_fca":"one_hot_zadeh_all",
         "code_width":10,
         "name":"ResNet20v2 - CRC7 - ZADEH",
         "index":11 },

        #SOFTMAX CRC7

        {"network_id": "cnn1_euclide_softmax",
         "network_activation":"l2_norm",
         "network_weights":"cnn1_crc7_softmax",
         "code_id":"crc7",
         "data_type":"euclide_softmax",
         "data_type_fca":"euclide_softmax_all",
         "code_width":10,
         "name":"CNN1 - CRC7 - SOFTMAX",
         "index":4 },

        {"network_id": "cnn2_euclide_softmax",
         "network_activation":"l2_norm",
         "network_weights":"cnn2_crc7_softmax",
         "code_id":"crc7",
         "data_type":"euclide_softmax",
         "data_type_fca":"euclide_softmax_all",
         "code_width":10,
         "name":"CNN2 - CRC7 - SOFTMAX",
         "index":10 },

        {"network_id": "ResNet20v2_euclide_softmax",
         "network_activation":"l2_norm",
         "network_weights":"resnet20v2_crc7_softmax",
         "code_id":"crc7",
         "data_type":"euclide_softmax",
         "data_type_fca":"euclide_softmax_all",
         "code_width":10,
         "name":"ResNet20v2 - CRC7 - SOFTMAX",
         "index":14 },

        #BIT THRESHOLD

        {"network_id": "cnn1_bit_threshold",
         "network_activation": None,
         "network_weights":"cnn1_crc7_bit_threshold",
         "code_id":"crc7",
         "data_type":"binary",
         "data_type_fca":"binary_all",
         "code_width":10,
         "name":"CNN1 - CRC7 - BIT_THRESHOLD",
         "index":5 },

         #HADAMARD EUCLIDE

        {"network_id": "cnn1_hadamard",
         "network_activation": None,
         "network_weights":"cnn1_hadamard_euclide",
         "code_id":"crc_hadamard",
         "data_type":"hadamard_euclide",
         "data_type_fca":"hadamard_euclide_all",
         "code_width":15,
         "name":"CNN1 - CRC_HADAMARD - EUCLID",
         "index":6 }

        ]

    path = os.path.dirname(os.path.realpath(__file__))

    dataset = data.Dataset()
    dataset.InitializeData()

    for p in parameters:

        print("%s"%p['name'])

        codePath = os.path.join(path,"codes","%s.csv"% p['code_id'])
        code = bincode.BINCode(width=p['code_width'],dataType=p['data_type'])
        code.LoadCode(codePath)

        print("  Creating new model: %s" %p['network_id'])
        model = networks.CreateNetwork(p['network_id'], dataset.ImageShape(), code, p['network_activation'] )

        weightsPath = os.path.join(path,"networks_weights","%s.hdf5"%p['network_weights'])
        print("  Loading network weights: %s"%weightsPath)

        model.load_weights(weightsPath)

        imgs,labels = dataset.TestData("cifar10")
        print("  Computing predictions")
        predictions = model.predict(imgs)

        acc = code.ComputePrecision(predictions,labels)
        p['accuracy'] = acc



    #
    for p in parameters:

        print("%s"%p['name'])

        p['data_type'] = p['data_type_fca']

        codePath = os.path.join(path,"codes","%s.csv"% p['code_id'])
        code = bincode.BINCode(width=p['code_width'],dataType=p['data_type'])
        code.LoadCode(codePath)

        print("  Creating new model: %s" %p['network_id'])
        model = networks.CreateNetwork(p['network_id'], dataset.ImageShape(), code, p['network_activation'] )

        weightsPath = os.path.join(path,"networks_weights","%s.hdf5"%p['network_weights'])
        print("  Loading network weights: %s"%weightsPath)

        model.load_weights(weightsPath)

        imgs,labels = dataset.TestData("cifar10")
        print("  Computing predictions")
        predictions = model.predict(imgs)

        p['fca'],p['undetectable'],p['reliability'] = code.ComputePrecision(predictions,labels)



    parameters = sorted(parameters,key=lambda x:x['index'])
    print("Table4: Performance of the various output coding and decision strategies.\n")
    print("n\tNetwork - Code - Decision\tAccuracy\tFCA\t\tUndetectable Error\tReliability\n")
    for p in parameters:
        if p['index'] == 7 or p['index'] == 11:
            print("")

        print("%2s %30s: \t%.2f\t\t%.2f\t\t\t%.2f\t\t\t%.2f"%(p['index'],p['name'],p['accuracy'],p['fca'],p['undetectable'],p['reliability']))



if __name__ == '__main__':
    main()