A set of modules to easily build a classifier in tensorflow

## **Modules**

The machine learning pipeline has been divided into **five** modules  

1. **loader.py** - loads the images files from a directory and creates a train and valid dataset.\
                    Load test function which load a directory of test files

2. **augopt.py** - returns a dataset with applied augmentation and various optimisation

3. **arch.py** - returns the model architecture

4. **fitcompile.py** - 
* compiles the model for loss, metrics and optimiser
* fit the model

5. **evaluate.py** - plots the confusion metrics and saves the missclassified images

6. **usage.py** - shows how to use the modules and set the hyperparameters                                           

Documentation URL : http://kailashsp.github.io/