from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.constraints import maxnorm


def cnn_model(input_shape, num_classes):
    """
    Implements Convolutional Neural Network with the following architecture:
    Sequential Model -> Conv2D with 32 filters -> Activation -> MaxPooling2D
                    -> Conv2D with 64 filters -> Activation -> MaxPooling2D
                    -> Conv2D with 128 filters -> Activation -> MaxPooling2D
                    -> Flatten -> Dense -> Activation -> Dropout -> Dense
    :param input_shape: height and width of images passed through the model
    :param num_classes: the number of classes
    :return: returns a keras model
    """

    # first stage: initialize sequential model and define input with input_shape
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, padding='same', name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    # second stage
    model.add(Conv2D(64, kernel_size=3, padding='same', name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    # third stage
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    # output layer
    model.add(Flatten())
    model.add(Dense(128, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    if num_classes > 2:
        model.add(Dense(num_classes, activation='softmax', name=f'fc_multi{num_classes}'))
    else:
        model.add(Dense(num_classes, activation='sigmoid', name=f'fc_binary{num_classes}'))

    return model
