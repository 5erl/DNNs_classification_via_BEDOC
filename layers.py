
import tensorflow as tf
import tensorflow.keras.layers as klayer
from tensorflow.python.keras import backend as K

import numpy as np
import bincode
import sys




def LOGPrint(tensor,text):
         tf.print(text)
         tf.print(tf.shape(tensor),summarize=-1,output_stream=sys.stdout)
         #tf.print(tf.shape(tensor),summarize=-1)
         tf.print (tensor,summarize=-1,output_stream=sys.stdout)



class LayerOneZeroNorm(klayer.Layer):
    def __init__(self,**kwargs):
        super(LayerOneZeroNorm, self).__init__(**kwargs)

    def call(self, inputs, training):

        minimum = tf.reduce_min(inputs,axis=-1)

        minimum_exp = tf.expand_dims(minimum,axis=-1)
        minimum_exp = tf.broadcast_to(minimum_exp, tf.shape(inputs))

        #LOGPrint(minimum_exp,"minimum_exp")
        non_neg = tf.where(tf.less(minimum_exp, 0.0), -minimum_exp, 0.0)
        #LOGPrint(non_neg,"non_neg")

        non_neg_input = inputs + non_neg

        #LOGPrint(non_neg_input,"non_neg_input")

        min_exp = tf.reduce_min(non_neg_input,axis=-1)
        min_exp = tf.expand_dims(min_exp,axis=-1)
        min_exp = tf.broadcast_to(min_exp, tf.shape(non_neg_input))

        #LOGPrint(min_exp,"min_exp")

        max_exp = tf.reduce_max(non_neg_input,axis=-1)
        max_exp = tf.expand_dims(max_exp,axis=-1)
        max_exp = tf.broadcast_to(max_exp, tf.shape(non_neg_input))

       # LOGPrint(max_exp,"max_exp")

        result = tf.math.divide(
           tf.math.subtract(
              non_neg_input,
              min_exp
           ),
           tf.math.subtract(
              max_exp,
              min_exp
           )
        )

        #LOGPrint(result,"norm_input")
        return result


class LayerL2Norm(klayer.Layer):
    def __init__(self,width,**kwargs):
        super(LayerL2Norm, self).__init__(**kwargs)
        self.width = width

    def call(self, inputs, training):

        #LOGPrint(inputs,"inputs 000")

        norm = tf.norm( inputs, ord=2,axis=-1)
        norm_expand = tf.expand_dims(norm,axis=-1)

        norm_expand = K.repeat_elements(norm_expand, rep=self.width, axis=-1)

        return ((inputs/norm_expand) + 1 )/ 2.0




class LayerZadeh2OneHot(klayer.Layer):

    def __init__(self, code,**kwargs):

        super(LayerZadeh2OneHot, self).__init__(**kwargs)


        dataType = code.GetDataType()
        #print("LayerZadeh2OneHot layer init: %s"% dataType )

        if dataType == "one_hot_zadeh":

            self.codes_width = code.CodeWidth()
            oneHotLabelAliases = code.CreateOneHotClasses()

            self.codes_count = code.TranslateTableAliasesCount()

            table = np.zeros((self.codes_count,self.codes_width),dtype=np.float32)

            index = self.codes_count - 1

            for item in oneHotLabelAliases:
                c = bincode.BINCode.Num2BinArray(item[1],self.codes_width,type=np.float32)
                #print("[%s] L:%s,C:%s -> %s -> %s" % (index,item[0],item[1],c,item[4]))
                table[index] = c
                index -= 1

        elif dataType == "one_hot_zadeh_all":

            #hardcoded!!!
            self.codes_width = 10
            self.codes_count = 1024

            table = np.zeros((self.codes_count,self.codes_width),dtype=np.float32)

            index = self.codes_count-1

            for code_index in range(self.codes_count):
                table[index] = code.Num2BinArray(code_index,self.codes_width,type=np.float32)

                #print("[%s] C:%s --> %s " % (index,code,table[index]))
                index -= 1
        else:
            print("ERROR: LayerZadeh2OneHot unsupported datatype %s !"%dataType)
            exit(1)

        self.codes_table = tf.convert_to_tensor(table, dtype=tf.float32)
        self.codes_table_expand = tf.expand_dims(self.codes_table,axis=0)

        

    def build( self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.codes_count)

    def manhattan_size(self,y_true, y_pred):
        return K.sum(K.abs(y_pred - y_true),axis=-1)


    def call(self, inputs, training):


        y_expand = tf.expand_dims(inputs,axis=-2)
      #  LOGPrint(y_expand,"y_expand")


        y_expand = K.repeat_elements(y_expand, rep=self.codes_count, axis=-2)
        #LOGPrint(y_expand,"y_expand")

        broadcast_shape = (tf.shape(inputs)[0], self.codes_count,self.codes_width)

        tables_expand = tf.broadcast_to(self.codes_table_expand, broadcast_shape)
        #LOGPrint(tables_expand,"tables_expand")


        together = tf.multiply(y_expand,tables_expand) + tf.multiply(1.0-y_expand,1.0-tables_expand)


        return tf.math.reduce_min(together,axis=-1)



class LayerEuclideSoftmax(klayer.Layer):

    def __init__(self, code,**kwargs):

        super(LayerEuclideSoftmax, self).__init__(**kwargs)


        dataType = code.GetDataType()
        #print("LayerEuclideSoftmax layer init: %s"% dataType )

        if dataType == "euclide_softmax" or dataType == "euclide_softmax_all":

            self.codes_width = code.CodeWidth()
            oneHotLabelAliases = code.CreateOneHotClasses()

            self.codes_count = code.TranslateTableAliasesCount()

            table = np.zeros((self.codes_count,self.codes_width),dtype=np.float32)

            index = self.codes_count - 1
            for item in oneHotLabelAliases:
                c = bincode.BINCode.Num2BinArray(item[1],self.codes_width,type=np.float32)
                #print("[%s] L:%s,C:%s -> %s -> %s" % (index,item[0],item[1],c,item[4]))
                table[index] = c
                index -= 1

        else:
            print("DataType must be euclide_sofmtax!")
            exit(1)

        self.codes_table = tf.convert_to_tensor(table, dtype=tf.float32)
        self.codes_table_expand = tf.expand_dims(self.codes_table,axis=0)


       # LOGPrint(self.codes_table,"self.codes_table")




    def build( self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.codes_count)

    def manhattan_size(self,y_true, y_pred):
        return K.sum(K.abs(y_pred - y_true),axis=-1)

    def euclide_size(self,y_true, y_pred):
        return K.pow(K.sum(K.pow(y_pred - y_true,2),axis=-1)+0.00000001,0.5)


    def call(self, inputs, training):



        #norms = tf.keras.utils.normalize(inputs,axis=-1,order=1)

        #LOGPrint(inputs,"inputs")
        y_expand = tf.expand_dims(inputs,axis=-2)
      #  LOGPrint(y_expand,"y_expand")


        y_expand = K.repeat_elements(y_expand, rep=self.codes_count, axis=-2)
        #LOGPrint(y_expand,"y_expand")


        broadcast_shape = (tf.shape(inputs)[0], self.codes_count,self.codes_width)

        tables_expand = tf.broadcast_to(self.codes_table_expand, broadcast_shape)
#        LOGPrint(tables_expand,"tables_expand")

        #LOGPrint(y_expand,"y_expand")

        euclide_y = self.euclide_size(y_expand,tables_expand)

        #LOGPrint(euclide_y,"LAYER ECOCLayerEuclideSoftmax: euclide_y")

        #negative euclide distance
        #should be max euclidean distance instead of 10.0 but for this purpose is enough
        euclide_y = 10.0 - euclide_y
        #LOGPrint(euclide_y,"LAYER ECOCLayerEuclideSoftmax: 10 - euclide_y")



        return  euclide_y

