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

    def __init__(self, model_name, image_size: List[int], cache: bool = True, prefetch: bool = True):

        self.cache = cache
        self.prefetch = prefetch
        self.image_size = image_size
        self.model_name = model_name


        self.rng = tf.random.Generator.from_seed(123, alg='philox')

    def normalize(self,image):
        '''returns images normalized or as desired 
        for pretrained model and resized'''
        
        # image = image / tf.constant(255, dtype=tf.float32)

        image = tf.image.resize(image,self.image_size,method='area')
        if 'xception' in self.model_name.lower():
            image = tf.keras.applications.xception.preprocess_input(image)
        elif 'efficientnet' in self.model_name.lower() and 'v2' in self.model_name.lower():
            image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        elif 'efficientnet' in self.model_name.lower():
            image = tf.keras.applications.efficientnet.preprocess_input(image)
        elif 'vgg16' in self.model_name.lower():
            image = tf.keras.applications.vgg16.preprocess_input(image)
        elif 'vgg19' in self.model_name.lower():
            image = tf.keras.applications.vgg19.preprocess_input(image)
        elif 'resnet' in self.model_name.lower() and 'v2' in self.model_name.lower():
            image = tf.keras.applications.res_v2.preprocess_input(image)
        elif 'resnet' in self.model_name.lower():
            image = tf.keras.applications.resnet.preprocess_input(image)
        elif 'mobilenet' in self.model_name.lower() and 'v3' in self.model_name.lower():
            return image
        elif 'mobilenet' in self.model_name.lower() and 'v2' in self.model_name.lower():
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        elif 'mobilenet' in self.model_name.lower():
            image = tf.keras.applications.mobilenet.preprocess_input(image)
        elif 'densenet' in self.model_name.lower():
            image = tf.keras.applications.densenet.preprocess_input(image)
        elif 'nasnet' in self.model_name.lower():
            image = tf.keras.applications.nasnet.preprocess_input(image)
        elif 'inception' in self.model_name.lower() and 'resnet' in self.model_name.lower():
            image = tf.keras.applications.inception_resnet_v2.preprocess_inpu(image)
        elif 'inception' in self.model_name.lower():
            image = tf.keras.applications.inception_v3.preprocess_input(image)

        return image

    # data augmentation applied on layers
    # resize_and_rescale = tf.keras.Sequential([
    # tf.keras.layers.Resizing(1024, 1024),
    # tf.keras.layers.Rescaling(1./255)
    # ])
    # the applied augmentation   
    # data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    # tf.keras.layers.RandomTranslation(0.3,0.3),
    # tf.keras.layers.RandomRotation(0.3),
    # tf.keras.layers.RandomContrast(0.8),
    # tf.keras.layers.RandomBrightness(0.7, value_range=(0, 255))
    # ])



    data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    ])
        

    
    def augmentation(self,image_label, seed):
        '''returns augmented images by randomly flipping them'''

        image, label = image_label
        # Make a new seed.
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

        seed1 = tf.random.experimental.stateless_split(new_seed, num=1)[0, :]

        # Random flip up and down.
        image = tf.image.stateless_random_flip_up_down(image, seed)
        # Random flip left and right
        image = tf.image.stateless_random_flip_left_right(image, new_seed)

        # image = tf.image.stateless_random_hue(image, 0.2, seed)

        # image = tf.image.stateless_random_saturation(image, 0.5, 1, seed1)

        # image = tf.image.stateless_random_contrast(image, 0.2, 0.5, seed)

        # image = tf.image.stateless_random_brightness(image, 0.2, new_seed)

        # image = tf.image.stateless_random_crop(image,seed,size=128)


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
            ds.shuffle(6000)

        if augment:
            ds = ds.map(lambda x,y: (self.data_augmentation(x,training=True),y),num_parallel_calls=tf.data.AUTOTUNE)
            # ds = ds.map(self.waug,num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(self.waug,num_parallel_calls=tf.data.AUTOTUNE)

        if self.prefetch:
            ds.prefetch(tf.data.AUTOTUNE)
        # manages the memory from being full
        # options = tf.data.Options()
        # options.threading.max_intra_op_parallelism = 1
        # ds = ds.with_options(options)
        return ds