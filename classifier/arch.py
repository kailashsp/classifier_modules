'''Creates a model'''

from typing import List, Tuple

import importlib
import tensorflow as tf



# #To make use of tensorcore in GPU
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

class Model:
    '''
    to define the model architecture
    returns the model as per user arguments and inputs'''

    def __init__(self, input_shape: Tuple[int], ndenselayers: List[int],prediction_classes,
                activation: str = 'relu' , dropout: float = 0.3, pretrained_model='MobileNetV3Large'):

        self.input_shape = input_shape
        self.ndenselayers = ndenselayers
        self.activation = activation
        self.dropout = dropout
        self.prediction_classes = prediction_classes
        self.pretrained_model = pretrained_model

    def import_model(self,modn):
        '''returns a pretrained model '''

        #User defined selection of pretrained model
        model_collection = importlib.import_module(name='tensorflow.keras.applications')
        return getattr(model_collection,modn)


    def dense_classifier(self, flattened, ndenselayers: List[int],
                         activation: str, dropout: float = 0.3):
        '''builds dense classifier layers on to pretrained model'''
   
        for i,layer in enumerate(ndenselayers):
            if i<1:
                x = tf.keras.layers.Dropout(dropout)(flattened)
                x = tf.keras.layers.Dense(layer,activation=activation)(x)
                
            else:
                x = tf.keras.layers.BatchNormalization()(x)     
                x = tf.keras.layers.Dense(layer,activation=activation)(x)
                x = tf.keras.layers.Dropout(dropout)(x)
              
               

        x = tf.keras.layers.Dense(self.prediction_classes,activation=tf.keras.activations.softmax)(x)
        output= tf.keras.layers.Activation('softmax',dtype='float32')(x)
        return output


    def arch(self):
        '''returns the model architecture with list of dense layers and the dropout value'''

        model_name = self.pretrained_model

        pretrained_model = self.import_model(model_name)

        conv_base = pretrained_model(weights="imagenet",include_top=False,input_shape=self.input_shape,pooling='avg')

        print(conv_base.summary())
   
        ltune = input("Enter the pretrained layer from which to unfreeze('None/none') for no fine tuning : ")

        conv_base.trainable = True
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == ltune:
                set_trainable = True
                print('fine tuning......')
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        input1 = tf.keras.Input(shape=self.input_shape)
        # x = tf.keras.applications.densenet.preprocess_input(input1)

        base = conv_base(input1)

        flat = tf.keras.layers.Flatten()(base)

        output = self.dense_classifier(flattened=flat, activation=self.activation,
                                            ndenselayers=self.ndenselayers,dropout=self.dropout)

        return tf.keras.Model(inputs = [input1], outputs = [output])
