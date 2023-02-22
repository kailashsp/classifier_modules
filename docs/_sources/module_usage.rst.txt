Module_Usage
============

Clone the directory 

Run the usage.py with the necessary changes. By default the pretrained model used is 
MobileNetV3LARGE 

.. note::
    Start with few layers, and create a decent baseline. 

    Keep the batch size constant.

    Changes in Optimisers / learning rate/ adding batch norm / adding drop out should be done step by step

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











