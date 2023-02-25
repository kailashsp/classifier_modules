Module_Usage
============

Steps to build a classifier

1. Clone the directory 

2. Ensure the images files follows the directory structure as given in :
https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

3. Run the usage.py with the necessary changes. By default the pretrained model used is 
MobileNetV3LARGE 

.. note::
    Start with few layers, and experiment to create a decent baseline. 

    Keep the batch size constant while changing other parameters.

    proceed with changes in Optimisers / learning rate/ adding batch norm / adding drop out 

    Use tensorboard to view the different model performance


Loading data
------------

.. autoclass:: classifier.loader.LoadDataset
    :members: load_train, load_valid


Augmentation and optimisation
-----------------------------

.. autoclass:: classifier.augopt.Mapping
    :members: normalize, augmentation, mapping

Architecture
------------

.. autoclass:: classifier.arch.Model
    :members: import_model,dense_classifier,arch

Training(compile the model and fitting)
---------------------------------------

.. autoclass:: classifier.fitcompile.Training
    :members: compile, fit

Evaluate
--------

.. note::
    Memory management issues on application to large datasets

.. autoclass:: classifier.evaluate.Evaluation
    :members: confusion_matrix, misclassified











