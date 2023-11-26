import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def plot_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(acc, label='Training acc')
    plt.plot(val_acc, label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig("accuracy_curve.png")

    # plt.show()

    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig("loss_curve.png")


def augmented_images(train_ds, batch_size):

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(batch_size):
         
            if not os.path.exists("augmented_images"):
                os.mkdir("augmented_images")

            img = Image.fromarray((np.squeeze(images[i].numpy())).astype(np.uint8))
            img.save(f"augmented_images/{i}.jpg")
            plt.title(int(labels[i]))
            plt.axis("off")


def cm_binary(valid_ds, model):
    '''plots the confusion matrix'''

    #creating empty 1D array to store labels from dataset and prediction
    val_labels = np.empty((0,3),int)
    pred = np.empty((0,3),int)

    # Iterating over individual batches to keep track of the images being fed to the model.
    for valid_images, valid_labels in valid_ds:
        
        val_labels = np.append(val_labels,valid_labels)

    # Model can take inputs other than dataset as well. 
    # Hence, after images are collected you can give them as input.
        y_val_pred = model.predict(valid_images)
        pred = np.append(pred,y_val_pred)

    return val_labels, [1 if p > 0.5 else 0 for p in pred]