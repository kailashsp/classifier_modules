'''
performs augmentation and optimisation on tf.data.Dataset

cache, shuffle, prefetch (for optimisation of dataset) and augment
are all boolean values stating whether the operation need be performed

Note : for validation or test dataset, shuffle and augment parameters should be set to False

'''

from typing import List

import tensorflow as tf

class Mapping:
    '''performs augmentation along with normalisation and rescaling,
    also optimisation of the tf.data dataset'''

    def __init__(self,image_size: List[int], cache: bool = True, prefetch: bool = True):

        self.cache = cache
        self.prefetch = prefetch
        self.image_size = image_size


        self.rng = tf.random.Generator.from_seed(123, alg='philox')

    def normalize(self,image):
        "returns images normalize [0,1] and resized"

        image = tf.image.resize(image,self.image_size,method='area')
        image = image / tf.constant(255, dtype=tf.float32)
        return image

    # data augmentation applied on layers
    # resize_and_rescale = tf.keras.Sequential([
    # tf.keras.layers.Resizing(1024, 1024),
    # tf.keras.layers.Rescaling(1./255)
    # ])
    # the applied augmentation   
    # data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("horizontal_and_vertical")
    # ])

    def augmentation(self,image_label, seed):
        '''returns augmented images by randomly flipping them'''

        image, label = image_label
        # Make a new seed.
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

        # Random flip up and down.
        image = tf.image.stateless_random_flip_up_down(image, seed)
        # Random flip left and right
        image = tf.image.stateless_random_flip_left_right(image, new_seed)

        return image, label

    # a wrapper function for updating seeds.
    def waug(self, x, y):
        '''returns images and labels'''

        seed = self.rng.make_seeds(2)[0]
        image, label = self.augmentation((x, y), seed)
        return image, label

    def mapping(self, ds, augment: bool = False, shuffle: bool = False):
        '''returns the images dataset normalised and  applying augmentation
        as the parameter set to the train dataset'''


        ds = ds.map(lambda x, y: (self.normalize(x), y),num_parallel_calls=tf.data.AUTOTUNE)

        if self.cache:
            ds.cache()

        if shuffle:
            ds.shuffle(1000)

        if augment:
            ds = ds.map(self.waug,num_parallel_calls=tf.data.AUTOTUNE)

        if self.prefetch:
            ds.prefetch(tf.data.AUTOTUNE)
        # manages the memory from being full
        # options = tf.data.Options()
        # options.threading.max_intra_op_parallelism = 1
        # ds = ds.with_options(options)
        return ds