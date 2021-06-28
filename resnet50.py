import keras
from keras.models import Model
from keras.layers import Add, Dense, Activation, ZeroPadding2D, BatchNormalization, \
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D


def identity_block(x, f, filters, stage, block):
    """
    Identity Block for ResNet

    Arguments:
    x - input tensor of shape (samples, height, width, channel)
    f - int, specifies shape of middle Conv2D layer
    filters - list of ints; number of filters for each CONV layer in the main path
    stage - int, used to name layers by position in network
    block - str, used to name layers by position in network

    Returns:
    x - output of identity block with shape (samples_new, height_new, width_new, channel_new)

    """

    # define names
    conv_name = f'res{stage}{block}'
    bn_name = f'bn{stage}{block}'

    # extract filters from list
    f1, f2, f3 = filters

    # save input value to add back to main path
    x_input = x

    # First block of main path
    x = Conv2D(f1, kernel_size=1, strides=1, padding='valid', name=conv_name + '2a')(x)
    x = BatchNormalization(name=bn_name + '2a')(x)
    x = Activation('relu')(x)

    # Second block of main path
    x = Conv2D(f2, kernel_size=f, strides=1, padding='same', name=conv_name + '2b')(x)
    x = BatchNormalization(name=bn_name + '2b')(x)
    x = Activation('relu')(x)

    # Third block of main path
    x = Conv2D(f3, kernel_size=1, strides=1, padding='valid', name=conv_name + '2c')(x)
    x = BatchNormalization(name=bn_name + '2c')(x)

    # input added back to main path and passed through a RELU activation
    x = Add()([x, x_input])
    x = Activation('relu')(x)

    return x


def convolutional_block(x, f, filters, stage, block, s=2):
    """
    Convolutional Block for ResNet

    Arguments:
    x - input tensor of shape (samples, height, width, channel)
    f - int, specifies shape of middle Conv2D layer
    filters - list of ints; number of filters for each CONV layer in the main path
    stage - int, used to name layers by position in network
    block - str, used to name layers by position in network

    Returns:
    x - output of identity block with shape (samples_new, height_new, width_new, channel_new)

    """

    # define names
    conv_name = f'res{stage}{block}'
    bn_name = f'bn{stage}{block}'

    # extract filters from list
    f1, f2, f3 = filters

    # save input value to add back to main path
    x_input = x

    # First block of main path
    x = Conv2D(f1, kernel_size=1, strides=s, name=conv_name + '2a')(x)
    x = BatchNormalization(name=bn_name + '2a')(x)
    x = Activation('relu')(x)

    # Second block of main path
    x = Conv2D(f2, kernel_size=f, strides=1, padding='same', name=conv_name + '2b')(x)
    x = BatchNormalization(name=bn_name + '2b')(x)
    x = Activation('relu')(x)

    # Third block of main path
    x = Conv2D(f3, kernel_size=1, strides=1, padding='valid', name=conv_name + '2c')(x)
    x = BatchNormalization(name=bn_name + '2c')(x)

    # skip connection path
    x_input = Conv2D(f3, kernel_size=1, strides=s, padding='valid', name=conv_name + '1')(x_input)
    x_input = BatchNormalization(axis=3, name=bn_name + '1')(x_input)

    # input added back to main path and passed through a RELU activation
    x = Add()([x, x_input])
    x = Activation('relu')(x)

    return x


def resnet50_model(input_shape, num_classes):
    """
    Implements ResNet50 with the following architecture:

    Conv2D -> BatchNormalization -> ReLu -> MaxPool -> Conv2D -> IDBlock*2
    -> ConvBlock -> IDBlock*3 -> ConvBlock -> IDBlock*5 -> ConvBlock
    -> IDBlock*2 -> AvgPool -> FCDenseLayer

    Arguments:
    input_shape - shape of images in the dataset
    classes - int, number of total classes

    Returns:
    model - returns a Keras model
    """

    # Define the input with shape input_shape
    x_input = keras.Input(shape=input_shape)

    # Zero-padding
    x = ZeroPadding2D(3)(x_input)

    # First Stage
    x = Conv2D(64, kernel_size=7, strides=2, name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2)(x)

    # Second Stage
    x = convolutional_block(x, f=3, filters=[64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # Third Stage
    x = convolutional_block(x, f=3, filters=[128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Fourth Stage
    x = convolutional_block(x, f=3, filters=[256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Fifth Stage
    x = convolutional_block(x, f=3, filters=[512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # output layer
    x = AveragePooling2D(2, name='avg_pool', padding='same')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name=f'fc{num_classes}')(x)

    rn50model = Model(inputs=x_input, outputs=x, name='ResNet50')

    return rn50model
