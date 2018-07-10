from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import warnings
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Activation, Lambda, Add, concatenate, Reshape
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import initializers, regularizers, constraints
from keras.utils import conv_utils, layer_utils
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions, _obtain_input_shape
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import keras

import numpy as np
from sklearn.cross_validation import train_test_split

# Own module
import sys
sys.path.append("../utils/")
import smoothL1, relu6
from layers import DepthwiseConv2D

INPUT_SHAPE = (64, 64, 1)
OUTPUT_SIZE = 136
N_LANDMARK = 68

def facial_landmark_cnn(input_shape=INPUT_SHAPE, output_size=OUTPUT_SIZE):
    # Stage 1 #
    img_input = Input(shape=input_shape)
    
    ## Block 1 ##
    x = Conv2D(32, (3,3), strides=(1,1), name='S1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu', name='S1_relu_conv1')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='S1_pool1')(x)

    ## Block 2 ##
    x = Conv2D(64, (3,3), strides=(1,1), name='S1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='S1_relu_conv2')(x)
    x = Conv2D(64, (3,3), strides=(1,1), name='S1_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='S1_relu_conv3')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='S1_pool2')(x)

    ## Block 3 ##
    x = Conv2D(64, (3,3), strides=(1,1), name='S1_conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='S1_relu_conv4')(x)
    x = Conv2D(64, (3,3), strides=(1,1), name='S1_conv5')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='S1_relu_conv5')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='S1_pool3')(x)
        
    ## Block 4 ##
    x = Conv2D(256, (3,3), strides=(1,1), name='S1_conv8')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='S1_relu_conv8')(x)
    x = Dropout(0.2)(x)
    
    ## Block 5 ##
    x = Flatten(name='S1_flatten')(x)
    x = Dense(2048, activation='relu', name='S1_fc1')(x)
    x = Dense(output_size, activation=None, name='S1_predictions')(x)
    model = Model([img_input], x, name='facial_landmark_model')
    
    return model

def main():
#        Define X and y
# #        Load data
        PATH = "./data/64_64_1/offset_1.3/"
        X = np.load(PATH + "basic_dataset_img.npz")
        y = np.load(PATH + "basic_dataset_pts.npz")
        X = X['arr_0']
        y = y['arr_0'].reshape(-1, 136)
        

        print("Define X and Y")
        print("=======================================")
        
        # Split train / test dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print("Success of getting train / test dataset")
        print("=======================================")
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        print("=======================================")

        model.compile(loss=smoothL1, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['mape'])
        print(model.summary())
        # checkpoint
        filepath="./basic_checkpoints/smooth_L1-{epoch:02d}-{val_mean_absolute_percentage_error:.5f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, batch_size=64, epochs=10000, shuffle=True,\
                            verbose=1, validation_data=(X_test, y_test), callbacks=callbacks_list)

        # Save model
        model.save("./model/face_landmark_dnn.h5")
        print("=======================================")
        print("Save Final Model")
        print("=======================================")
        
if __name__ == "__main__":
    main()
