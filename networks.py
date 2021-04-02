
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as kl
import tensorflow.keras.layers as ka
import tensorflow.keras.models as km
import tensorflow as tf

from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Lambda,AveragePooling2D,Input,GlobalAveragePooling2D

import tensorflow.keras.optimizers as ko

import layers as ecl


def CreateNetwork(netID,img_shape,code,activation):

    label_width = code.CodeWidth()

    if netID == "cnn1_onehot" or netID == "cnn1_bit_threshold" or netID == "cnn1_hadamard":

        weight_decay = 1e-4
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=img_shape))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(label_width))

        if activation == "one_zero_norm":
            model.add(ecl.LayerOneZeroNorm())
        elif activation == "l2_norm":
            model.add(ecl.LayerL2Norm(label_width))
        else:
            model.add(Activation(activation))


    elif netID == "cnn1_zadeh":


        weight_decay = 1e-4
        model = Sequential()

        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=img_shape))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(label_width))

        if activation == "one_zero_norm":
            model.add(ecl.LayerOneZeroNorm())
        elif activation == "l2_norm":
            model.add(ecl.LayerL2Norm(label_width))
        else:
            model.add(Activation(activation))

        model.add(ecl.LayerZadeh2OneHot(code=code))


    elif netID == "cnn1_euclide_softmax":


        weight_decay = 1e-4
        model = Sequential()

        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=img_shape))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        #model.add(Dense(label_width,activation=activation))
        model.add(Dense(label_width))
        #model.add(BatchNormalization())
        if activation == "one_zero_norm":
            model.add(ecl.LayerOneZeroNorm())
        elif activation == "l2_norm":
            model.add(ecl.LayerL2Norm(label_width))
        else:
            model.add(Activation(activation))

        model.add(ecl.LayerEuclideSoftmax(code=code))
        model.add(Activation("softmax"))


    elif netID == "cnn2_onehot":


        weight_decay = 1e-4
        model = Sequential()
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=img_shape))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(label_width,activation=activation))


    elif netID == "cnn2_zadeh":

        weight_decay = 1e-4
        model = Sequential()
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=img_shape))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(label_width))
        if activation == "one_zero_norm":
            model.add(ecl.LayerOneZeroNorm())
        elif activation == "l2_norm":
            model.add(ecl.LayerL2Norm(label_width))
        else:
            model.add(Activation(activation))

        model.add(ecl.LayerZadeh2OneHot(code=code))

    elif netID == "cnn2_euclide_softmax":


        weight_decay = 1e-4
        model = Sequential()
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=img_shape))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3),  padding='valid',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(label_width))
        if activation == "one_zero_norm":
            model.add(ecl.LayerOneZeroNorm())
        elif activation == "l2_norm":
            model.add(ecl.LayerL2Norm(label_width))
        else:
            model.add(Activation(activation))

        model.add(ecl.LayerEuclideSoftmax(code=code))
        model.add(Activation("softmax"))



    elif netID == "ResNet20v2_onehot":

        depth = 2 * 9 + 2
        model = resnet_v2(input_shape=img_shape, depth=depth, num_classes=label_width, activation_dense=activation,onehot=True)

    elif netID == "ResNet20v2_zadeh":

        depth = 2 * 9 + 2
        model = resnet_v2(input_shape=img_shape, depth=depth, num_classes=label_width, activation_dense=activation,
                          onehot=False,code=code)

    elif netID == "ResNet20v2_euclide_softmax":

        depth = 2 * 9 + 2
        model = resnet_v2(input_shape=img_shape, depth=depth, num_classes=label_width, activation_dense=activation,
                          onehot=False,code=code,euc_soft=True)

    else:
        print ("ERROR: Unknown network! ", netID)
        exit(1)


    model.compile( optimizer = ko.Adam(lr=0.001))
    #model.summary()

    return model




def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=10,activation_dense=None,onehot=False,code=None,
              globalActivation="relu",euc_soft=False):


    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = globalActivation
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = kl.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation(globalActivation)(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    if onehot == True:
        outputs = Dense(num_classes,
                        activation=activation_dense,
                        kernel_initializer='he_normal')(y)

    else:
        outputs = Dense(num_classes,
                        activation = None,
                        kernel_initializer='he_normal')(y)

        if activation_dense == "one_zero_norm":
            outputs = ecl.LayerOneZeroNorm()(outputs)
        elif activation_dense == "l2_norm":
            outputs = ecl.LayerL2Norm(num_classes)(outputs)
        else:
            outputs = Activation(activation_dense)(outputs)

        if euc_soft == True:
            outputs = ecl.LayerEuclideSoftmax(code=code)(outputs)
        else:
            outputs = ecl.LayerZadeh2OneHot(code=code)(outputs)


    # Instantiate model.
    model = km.Model(inputs=inputs, outputs=outputs)
    return model



