'''
Loads the data as tf.data.dataset format with the parameters of filepath, batch and validation split

returns a train dataset/validation dataset on calling functions load_train() or load_valid()
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import tensorflow as tf

class LoadDataset:
    '''loads the data from directory as tf.data.dataset for train and valid dataset creation'''

    def __init__(self, fpath,image_size, batch_size: int, split: float = 0):

        self.fpath = fpath
        self.split = split
        self.batch_size = batch_size
        self.image_size = image_size


    def load_train(self):
        '''returns a train dataset based on the parameter inputs'''

        return tf.keras.utils.image_dataset_from_directory(self.fpath, subset="training",
                                                            validation_split=self.split, interpolation='area',
                                                            seed=42, batch_size= self.batch_size,
                                                            image_size=self.image_size, label_mode='int')

    def load_valid(self):
        '''returns a valid dataset based on the parameter inputs'''

        return tf.keras.utils.image_dataset_from_directory(self.fpath, subset="validation",interpolation='area',
                                                               validation_split=self.split,
                                                               seed=42, batch_size=self.batch_size,
                                                               image_size=self.image_size, label_mode='int')

    
    # def load_test(self,filepath):
    #     '''returns a test dataset based on the parameter inputs'''

    #     return tf.keras.utils.image_dataset_from_directory(filepath,seed=42, batch_size=self.batch_size,
    #                                                            image_size=self.image_size, label_mode='categorical',
    #                                                            shuffle=False)
      