import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/home/kailash/anaconda3/envs/Classifier_env/lib'

os.environ['AWS_ACCESS_KEY_ID']='AKIA52TAZEDD2NZ4GNUJ'
   # The access key for your AWS account.
os.environ['AWS_SECRET_ACCESS_KEY']='VAD7A1chOi4pm+JcYHENW9K5ROHU+GPUvDciBDTo'
   # The secret access key for your AWS account.
import boto3

my_session = boto3.session.Session()
my_region = my_session.region_name
print(my_region)
import numpy as np
import pandas as pd
import argparse
from classifier.loader import LoadDataset
from classifier.augopt import Mapping
from classifier.arch import Model
from classifier.evaluate import Evaluation
from classifier.fitcompile import Training
import matplotlib.pyplot as plt
import subprocess
#export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
#export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
#check whether GPU is being used
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

import mlflow
from mlflow.models.signature import infer_signature

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
#check whether GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


parser =  argparse.ArgumentParser()

parser.add_argument("-pm","--pretrained_model",
                    help="one of the pretrained models from keras.applications",
                    default='MobileNetV3Large')

args = vars(parser.parse_args())
trained_model = args["pretrained_model"]


#keep the training images in Dataset folder
l = LoadDataset(fpath='Datasets/Kitchenwares',
                batch_size = 32,
                split=0.2,
                image_size=[400,400])

train_ds = l.load_train()

valid_ds = l.load_valid()

print(train_ds.class_names)

# test_ds = l.load_test('augmented_images')

m = Mapping(image_size=[224,224],
            prefetch=False,model_name = trained_model)

train_ds = m.mapping(ds =train_ds,
                     augment=True,
                    shuffle=False)

valid_ds = m.mapping(ds = valid_ds, 
                    augment=False,
                    shuffle=False)

# check the augmented images
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(4):
#     if not os.path.exists("augmented_images"):
#       os.mkdir("augmented_images")

#     tf.keras.utils.save_img(f"augmented_images/image{i}.png", images[i])


#Defining model architecture
#for a custom architecture load the model and from the python file and continue

m = Model(input_shape=(224,224,3),
            ndenselayers=[64,32,16],
            activation = 'relu', 
            dropout=0.2,
            prediction_classes =6,
            pretrained_model=trained_model)

model = m.arch()



# model.summary()


md = Training(model=model)


track ="ec2-13-232-120-157.ap-south-1.compute.amazonaws.com"
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri(f"http://{track}:5000")
print(f"track uri:{mlflow.get_tracking_uri()}")


mlflow.set_experiment("Kitch_classification")

with mlflow.start_run():
   mlflow.set_tag("ml_engineer","kailash")

   opt = 'Adam'
   lr= 0.001
   epoch =25
   mlflow.log_params({"learning_rate":lr,
                     "epoch":epoch,
                     "optimiser":opt}
   )


   md.compile(loss='sparse_categorical_crossentropy',optimizer=opt,lrate = lr, metrics='accuracy')
   his,model = md.fit(epoch,train_ds=train_ds,valid_ds=valid_ds,model_name=trained_model)


   # mlflow.log_params(model.history
   # )

   #mlflow.log_artifact(f"models/{trained_model}.h5",artifact_path="models_tensorflow")
   # model_signature = infer_signature(train_ds,model.predict(train_ds))
   mlflow.tensorflow.log_model(model,artifact_path="models")

   mlflow.log_metrics({
                        "loss":min(his.history['val_loss']) }           
                           )


# opt = 'Adam'
# lr= 0.0001
# epoch =5
# mlflow.tensorflow.autolog()
# mlflow.ten

# md.compile(loss='sparse_categorical_crossentropy',optimizer=opt,lrate = lr, metrics='accuracy')
# metrics = md.fit(epoch,train_ds=train_ds,valid_ds=valid_ds)












# print('The accuracy of model  :',metrics.accuracy)
# print('The precision of model :',metrics.precision)
# print('The recall of model    :',metrics.recall)


# e = Evaluation(model,valid_ds)

# e.confusion_matrix()

# e.misclassified()
