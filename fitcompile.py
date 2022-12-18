''' compiles and trains the model
 and returns metrics(accuracy,precision,recall)'''

from collections import namedtuple

import os
import importlib

import numpy as np
import tensorflow as tf



class Training():
    ''' to invoke functions for compiling the model and training'''
    def __init__(self, model, model_checkpointpath : str):

        self.model = model
        self.model_checkpointpath = model_checkpointpath

    def compile(self, loss, optimizer:str = 'SGD', lrate:float = 0.0001, metrics:str = 'acc'):
        '''to compile the model for loss ,optimizer and metrics'''

        model_collection = importlib.import_module(name='tensorflow.keras.optimizers')
        opt = getattr(model_collection,optimizer)
        self.model.compile(loss=loss, optimizer=opt(learning_rate=lrate), metrics=metrics)

    def fit(self, epoch: int, train_ds, valid_ds):
        '''Trains the model, logs into tensorboard and
        returns accuracy, precision and recall as named tuple'''

        if not os.path.isdir('Logs'):
            os.mkdir('Logs')

        #logging data into tensorboard
        log = os.path.join(os.getcwd(),'''Logs/test''')
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log, profile_batch=(1,4))

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(self.model_checkpointpath,save_best_only=True)

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=2,restore_best_weights=True)

        self.model.fit(train_ds, epochs=epoch, validation_data=valid_ds,
                       callbacks=[tensorboard_cb,checkpoint_cb,early_stopping_cb])

        self.model = tf.keras.models.load_model(self.model_checkpointpath)

        
        #metrics for the model
        precision = tf.keras.metrics.Precision()
        accuracy = tf.keras.metrics.Accuracy()
        recall = tf.keras.metrics.Recall()

        # Iterating over individual batches to keep track of the images
        # being fed to the model.
        for valid_images, valid_labels in valid_ds:
            y_val_true = np.argmax(valid_labels, axis=1)

        # Model can take inputs other than dataset as well. Hence, after images
        # are collected you can give them as input.
            y_val_pred = self.model.predict(valid_images)
            y_val_pred = np.argmax(y_val_pred, axis=1)   
        # Update the state of the accuracy metric after every batch
            accuracy.update_state(y_val_true, y_val_pred)
            precision.update_state(y_val_true, y_val_pred)
            recall.update_state(y_val_true, y_val_pred)

        mm = namedtuple('metrics',['accuracy', 'precision', 'recall'])
             
        # print(f'Accuracy : {accuracy.result().numpy()}')
        # print(f'Precision : {precision.result().numpy()}')
        # print(f'Recall : {recall.result().numpy()}')

        return mm(accuracy.result().numpy(),
                  precision.result().numpy(),
                  recall.result().numpy())
