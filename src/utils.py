import sys
import os
import numpy as np

'''
Generator class for clothing segmentation dataset
Reference to : https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation for details on dataset
'''
import tensorflow as tf

def hello(name):
    return f"Hello world automatic update  {name}!"

#Apply random crop to picture
def random_crop(image, label):
    cropped_images = tf.image.random_crop(image, size=[256, 256, 3])
    cropped_images = tf.image.per_image_standardization(image)

    return cropped_images