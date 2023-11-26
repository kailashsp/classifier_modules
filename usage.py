import os
import glob
import argparse
import mlflow
from classifier.loader import LoadDataset
from classifier.augopt import Mapping
from classifier.arch import Model
from classifier.fitcompile import Training
import tensorflow as tf
from classifier.utils import plot_loss, augmented_images, cm_binary
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

# check whether GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


parser = argparse.ArgumentParser()

parser.add_argument("-pm", "--pretrained_model",
                    help="one of the pretrained models from keras.applications",
                    default='MobileNetV3Large')

parser.add_argument("-rn", "--run_name",
                    help="name to be assigned to mlflow run",
                    required=True)

parser.add_argument("-v", "--version",
                    help="assign the version number",
                    required=True)

args = vars(parser.parse_args())
trained_model = args["pretrained_model"]
run_name = args["run_name"]
version = args["version"]
batch_size = 2
image_height, image_width = (1024, 1024)
channels = 3

# keep the training images in Dataset folder
load_data = LoadDataset(fpath='Dataset/dataset',
                        batch_size=2,
                        split=0.2,
                        image_size=[image_height, image_width])

train_ds = load_data.load_train()

valid_ds = load_data.load_valid()

print(train_ds.class_names)

augmented_images(train_ds, batch_size=batch_size)

# test_ds = l.load_test('augmented_images')

m = Mapping(image_size=[image_height, image_width],
            prefetch=False, model_name=trained_model)

train_ds = m.mapping(ds=train_ds,
                     augment=False,
                     shuffle=False)

valid_ds = m.mapping(ds=valid_ds, 
                     augment=False,
                     shuffle=False)

ndenselayers = [16]
m = Model(input_shape=(image_height, image_width, channels),
          ndenselayers=ndenselayers,
          activation='relu',
          dropout=0.2,
          prediction_classes=1,
          pretrained_model=trained_model)

model = m.arch()

mod = Training(model=model)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print(f"track uri:{mlflow.get_tracking_uri()}")


mlflow.set_experiment("First_page_classification")

with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("ml_engineer", "kailash")
    opt = 'Adam'
    lr = 0.00001
    epoch = 50
    mlflow.log_params({"image_size": (image_height, image_width, channels),
                       "batch_size": batch_size,
                       "cnn_backbone": trained_model,
                       "dense_layers": ndenselayers
                       })

    mlflow.log_params({"learning_rate": lr,
                       "epoch": epoch,
                       "optimiser": opt})

    mod.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, 
                lrate=lr, metrics='accuracy')
  
    his, model = mod.fit(epoch, train_ds=train_ds,
                         valid_ds=valid_ds,
                         model_name=trained_model+"_V"+version)
    # mlflow.tensorflow.log_model(model,
    #                             artifact_path="model")
    tf.keras.utils.plot_model(model, show_shapes=True,
                              to_file="model.png")
    mlflow.log_artifact("model.png")
    mlflow.log_metrics({"val_loss": min(his.history['val_loss']),
                        "val_accuracy": min(his.history['val_accuracy']),
                        "train_loss": min(his.history['loss']),
                        "train_accuracy": min(his.history['accuracy'])
                        })

    plot_loss(his)
    for file in glob.glob("augmented_images"):
        mlflow.log_artifact(file)
    mlflow.log_artifact("accuracy_curve.png")
    mlflow.log_artifact("loss_curve.png")

    val_labels, pred = cm_binary(valid_ds, model)
    cm = confusion_matrix(val_labels, pred)
    result = cm.ravel()
    print(result)
    if len(result) == 4:
        t_n, f_p, f_n, t_p = result

        mlflow.log_metric("tn", t_n)
        mlflow.log_metric("fp", f_p)
        mlflow.log_metric("fn", f_n)
        mlflow.log_metric("tp", t_p)
    else:
        mlflow.log_metric("cm", result)

    disp = ConfusionMatrixDisplay(confusion_matrix(val_labels, pred))
    disp.plot().figure_.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_dict(classification_report(val_labels,
                                          pred, output_dict=True),
                                          "classification_report.json")