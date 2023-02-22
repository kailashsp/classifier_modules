'''
Class takes the validation/test dataset as input

the function takes model object and predicts the misclassified labels

and plots a confusion matrix and misclassified images to save to directory
'''
from time import perf_counter
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report

import os
import numpy as np
import tensorflow as tf


class Evaluation():
    '''view confusion matrix, classification report, and
    misclassified images saved to a folder'''

    def __init__(self,model,valid_ds):
        self.model = model
        self.valid_ds = valid_ds

    
    def confusion_matrix(self):
        '''plots the confusion matrix'''

        #creating empty 1D array to store labels from dataset and prediction
        val_labels = np.empty((0,3),int)
        pred = np.empty((0,3),int)

        start = perf_counter()
        # Iterating over individual batches to keep track of the images being fed to the model.
        for valid_images, valid_labels in self.valid_ds:
            y_val_true = np.argmax(valid_labels, axis=1)
            val_labels = np.append(val_labels,y_val_true)

        # Model can take inputs other than dataset as well. 
        # Hence, after images are collected you can give them as input.
            y_val_pred = self.model.predict(valid_images)
            y_val_pred = np.argmax(y_val_pred, axis=1)
            pred = np.append(pred,y_val_pred)
        end = perf_counter()
        print(f"Elapsed time for prediction : {end-start}\n\n")

        print(f'''{confusion_matrix(val_labels,pred)}\n\n''')
        
        print(f'''{classification_report(val_labels,pred)}\n\n''')
        disp = ConfusionMatrixDisplay(confusion_matrix(val_labels,pred))
        disp.plot()


    def misclassified(self):
        '''prints the numbers of misclassified images and saves them to a folder
        with their respective prediction and validation labels'''

        #creating empty 1D array to store labels from dataset and prediction
        val_labels = np.empty((0,3),int)
        pred = np.empty((0,3),int)
        valid_img =[]

        # Iterating over individual batches to keep track of the images being fed to the model.
        for valid_images, valid_labels in self.valid_ds:
            y_val_true = np.argmax(valid_labels, axis=1)
            val_labels = np.append(val_labels,y_val_true)
            valid_img.append(valid_images.numpy())
        # Model can take inputs other than dataset as well. 
        # Hence, after images are collected you can give them as input.
            y_val_pred = self.model.predict(valid_images)
            y_val_pred = np.argmax(y_val_pred, axis=1)
            pred = np.append(pred,y_val_pred)

        val_images = np.concatenate(valid_img,axis=0)

        #compare the two 1D arrays from test dataset and prediction to give the incorrect labels
        incorrect = np.nonzero(pred!= val_labels)
       # returns as tuple hence accessing first element to get 1D array
        misclass = incorrect[0]

        class_names = ['clean_images', 'letter_press', 'low_ink']

        print(f'''The number of misclassified labels: {len(misclass)} \n\n''')
 
        if not os.path.isdir('validation_misclassified'):
            os.mkdir('validation_misclassified')

        for i,arr in enumerate(misclass):
            path = f'''validation_misclassified/Prediction:{class_names[pred[arr]]} True:{class_names[val_labels[arr]]}_{i}.png'''
            tf.keras.utils.save_img(path, val_images[arr])