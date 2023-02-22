''' compiles and trains the model
 and returns metrics(accuracy,precision,recall)'''

from collections import namedtuple

import os
import glob
import importlib
import shutil

import numpy as np
import tensorflow as tf

import mlflow

class Training():
    ''' to invoke functions for compiling the model and training'''
    def __init__(self, model):

        self.model = model

    def compile(self, loss, optimizer:str = 'SGD', momentum:float =0.9,
                lrate:float = 0.0001, beta_1:float = 0.9, epsilon:float =1e-07,
                beta_2:float = 0.999,  metrics:str = 'acc'):

        '''to compile the model for loss ,optimizer and metrics'''

        model_collection = importlib.import_module(name='tensorflow.keras.optimizers')
        opt = getattr(model_collection,optimizer)

        if optimizer == 'Adam':
            optim = opt(learning_rate=lrate, beta_1= beta_1, beta_2=beta_2, epsilon=epsilon)  
        elif optimizer == 'AdamW':
            optim = opt(learning_rate=lrate, beta_1= beta_1, beta_2=beta_2, epsilon=epsilon)
        elif optimizer =='Nadam':
            optim = opt(learning_rate=lrate, beta_1= beta_1, beta_2=beta_2, epsilon=epsilon)

        elif optimizer == 'RMSprop':
            optim = opt(learning_rate=lrate, momentum=momentum)
        elif optimizer == 'SGD':
            optim = opt(learning_rate=lrate, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {optimizer} not implemented.')

        self.model.compile(loss=loss, optimizer=optim, metrics=metrics)



    def fit(self, epoch: int, train_ds, valid_ds,model_name):
        '''Trains the model, logs into tensorboard and
        returns accuracy, precision and recall as named tuple'''

        if os.path.exists('Logs'):
            sorted_by_time = sorted(glob.glob('Logs/*'), key=lambda t: os.stat(t).st_mtime)

        if len(glob.glob('Logs/*')) > 5:
            shutil.rmtree(sorted_by_time[-1])

        if not os.path.isdir('Logs'):
            os.mkdir('Logs')

        #logging data into tensorboard

        log = os.path.join(os.getcwd(),f'''Logs/{model_name}''')
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log, profile_batch=(1,4))

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(f"models/{model_name}.h5",
                                                            save_best_only=True,monitor="val_loss")

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                            restore_best_weights=True)

        history = self.model.fit(train_ds, epochs=epoch, validation_data=valid_ds,
                       callbacks=[tensorboard_cb,checkpoint_cb,early_stopping_cb])
        
        # mlflow.tensorflow.log_model(model=self.model, artifact_path="model")



        self.model = tf.keras.models.load_model(f"models/{model_name}.h5")

        return history,self.model
        
        #metrics for the model
        # precision = tf.keras.metrics.Precision()
        # accuracy = tf.keras.metrics.Accuracy()
        # recall = tf.keras.metrics.Recall()

        # Iterating over individual batches to keep track of the images
        # # being fed to the model.
        # for valid_images, valid_labels in valid_ds:
        #     y_val_true = np.argmax(valid_labels, axis=1)

        # # Model can take inputs other than dataset as well. Hence, after images
        # # are collected you can give them as input.
        #     y_val_pred = self.model.predict(valid_images)
        #     y_val_pred = np.argmax(y_val_pred, axis=1)   
        # # Update the state of the accuracy metric after every batch
        #     accuracy.update_state(y_val_true, y_val_pred)
        #     precision.update_state(y_val_true, y_val_pred)
        #     recall.update_state(y_val_true, y_val_pred)

        # mm = namedtuple('metrics',['accuracy', 'precision', 'recall'])
             
        # print(f'Accuracy : {accuracy.result().numpy()}')
        # print(f'Precision : {precision.result().numpy()}')
        # print(f'Recall : {recall.result().numpy()}')

        # return mm(accuracy.result().numpy(),
        #           precision.result().numpy(),
        #           recall.result().numpy())
