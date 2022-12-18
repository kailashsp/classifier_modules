'''Creates a model based on given the input shape 
ndenselayers is a list of number of neurons in each dense classifier layer
activation 
dropout value
dropout state
batchnorm
'''

from typing import List, Tuple

import importlib
import tensorflow as tf



# #To make use of tensorcore in GPU
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

class Model:
    '''to define the model architecture
    returns the model as pe user arguments and inputs'''

    def __init__(self, input_shape: Tuple[int], predicted_classes, ndenselayers: List[int],
                activation: str = 'relu' , dropout: float = 0.3, dropout_state: bool = False, batch_norm: bool = False):

        self.input_shape = input_shape
        self.ndenselayers = ndenselayers
        self.activation = activation
        self.dropout = dropout
        self.prediction_classes = predicted_classes
        self.dropout_state = dropout_state
        self.batch_norm = batch_norm

    def import_model(self,modn):
        '''returns a pretrained model '''

        #User defined selection of pretrained model
        model_collection = importlib.import_module(name='tensorflow.keras.applications')
        return getattr(model_collection,modn)


    def dense_classifier(self, flattened, ndenselayers: List[int],
                         activation: str, dropout: float = 0.3):
        '''returns a model with a dense classifier'''

        x = flattened
        for i,layer in enumerate(ndenselayers):
            if i<1:
                if self.dropout_state:
                    x = tf.keras.layers.Dropout(dropout)(x)
                x = tf.keras.layers.Dense(layer,activation=activation)(x)
                
            else:
                if self.batch_norm:
                    x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dense(layer,activation=activation)(x)
                x = tf.keras.layers.Dropout(dropout)(x)
               

        x = tf.keras.layers.Dense(self.prediction_classes,activation=tf.keras.activations.softmax)(x)
        output= tf.keras.layers.Activation('softmax',dtype='float32')(x)
        return output


    def arch(self):
        '''returns the model architecture accepts a list of dense layers and the dropout value'''
        

        model_name = input('Enter the pretrained model: ')

        pretrained_model = self.import_model(model_name)

        conv_base = pretrained_model(weights="imagenet",include_top=False,input_shape=self.input_shape)

        print(conv_base.summary())

        conv_base.trainable = True
        
        print("Number of layers in the base model: ", len(conv_base.layers))

        # # Fine-tune from this layer onwards
        # fine_tune_at = int(input("Enter the layer number from which to fine tune : "))

        # # Freeze all the layers before the `fine_tune_at` layer
        # for layer in conv_base.layers[:fine_tune_at]:
        #     layer.trainable = False
   
        ltune = input("Enter the pretrained layer from which to unfreeze('None/none' for no fine tuning): ")

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

        base = conv_base(input1)

        flat = tf.keras.layers.Flatten()(base)

        output = self.dense_classifier(flattened=flat, activation=self.activation,
                                            ndenselayers=self.ndenselayers,dropout=self.dropout)

        return tf.keras.Model(inputs = [input1], outputs = [output])
