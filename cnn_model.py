from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.constraints import maxnorm

# sequential model for CNN
def cnn_model(input_shape, num_classes):
    '''
    Implements Convolutional Neural Network with the following architecture:
    Sequential Model -> Conv2D -> BatchNormalization -> Conv2D
    -> MaxPooling2D -> BatchNormalization -> Conv2D -> BatchNormalization
    -> Flatten -> Dropout -> Dense -> BatchNormalization -> Dense ->
    BatchNormalization -> Dense
    :param input_shape: height and width of images passed through the model
    :param num_classes: the number of classes
    :return: returns a keras model
    '''

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    #model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.2))
    #model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    #model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_constraint=maxnorm(3), activation='relu'))
    #model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(128, kernel_constraint=maxnorm(3), activation='relu'))
    model.add(Dropout(0.2))
    #model.add(BatchNormalization())

    model.add(Dense(num_classes, activation='softmax'))

    return model