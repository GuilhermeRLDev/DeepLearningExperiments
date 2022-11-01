import sys
import os
import subprocess

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

def run_command(command, message = None):
    PIPE = subprocess.PIPE
    arr_command = command.split()
    print(arr_command)
    if message is not None:
        arr_command.append(message)

    process = subprocess.Popen(arr_command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    if stderr is not None and stderr != "":
        print(stderr)
    else:
        print(stdout)