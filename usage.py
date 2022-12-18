import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt

#custom modules
from loader import LoadDataset
from augopt import Mapping
from arch import Model
from evaluate import Evaluation
from fitcompile import Training


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
#check whether GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#loading the dataset 
l = LoadDataset(fpath='augmented_images1',batch_size = 4,split=0.2,image_size=[2048,2048])

train_ds = l.load_train()

valid_ds = l.load_valid()

#test_ds = l.load_test('augmented_images')

#for preprocessing optimizing and augmentation
m = Mapping(image_size=[512,512],cache=True,prefetch=True)

train_ds = m.mapping(ds =train_ds, augment=True,shuffle=True)

valid_ds = m.mapping(ds = valid_ds, augment=False,shuffle=False)

#check the augmented images obtained 
#which are save to the path specified
for images, labels in train_ds.take(1):
  for i in range(4):
    if not os.path.exists("augmented_images"):
      os.mkdir("augmented_images")

    tf.keras.utils.save_img(f"augmented_images/image{i}.png", images[i])

#Defining model architecture 
m = Model(input_shape=(512,512,3),ndenselayers=[32],activation = 'relu', dropout=0.3, predicted_classes=3,dropout_state=True,batch_norm=False)

model = m.arch()

model.summary()

#input the model build using arch module and the path to save the model
md = Training(model=model, model_checkpointpath="DENSENET121")

md.compile(loss='categorical_crossentropy',optimizer='SGD',lrate=0.0002,metrics='accuracy')

metrics = md.fit(10,train_ds=train_ds,valid_ds=valid_ds)

print('The accuracy of model  :',metrics.accuracy)
print('The precision of model :',metrics.precision)
print('The recall of model    :',metrics.recall)


#these can be used separately input a model loaded 
#and a tf.data.dataset which can be created using loader module
e = Evaluation(model,valid_ds)

e.confusion_matrix()

e.misclassified()
