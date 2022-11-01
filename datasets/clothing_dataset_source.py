import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as im
import numpy as np

class Dataset:
    # Init dataset by populating list of items and list of masks
    def __init__(self, location, masks_location, percentage=0.1):
        self.location = location
        self.files = os.listdir(location)
        self.files.remove(masks_location)

        self.masks = {}
        self.files = np.array(self.files)

        for file in self.files:
            if file not in self.masks:
                self.masks[file] = self.get_mask_path(location, masks_location, file)

        self.build_training_dataset(percentage)
        self.build_validation_dataset(percentage)

    def build_training_dataset(self, percentage):
        self.training_size = round(len(self.files) * (percentage))
        size = len(self.files) - self.training_size
        self.training = self.files[0:size]

    def build_validation_dataset(self, percentage):
        self.validation_size = round(len(self.files) * percentage)
        size = len(self.files) - self.validation_size
        self.validation = self.files[size:]

    # Return the masks location
    def get_mask_path(self, location, masks_location, file):
        file_name = file.split('_')[1]

        return f"{location}/{masks_location}/seg_{file_name}"

    # Load image from physical path
    def load_image(self, image, location, augment=True):

        if location is None:
            image = im.imread(image)
        else:
            image = im.imread(f"{location}/{image}")

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        return image

        # Display image

    def show_image(self, image):
        plt.imshow(image, cmap='gray')

    def get_batch(self, size):
        indexes = np.random.randint(0, len(self.files), size)

        # Get images from batch
        batch = self.files[indexes]
        batch_mask = np.array([])

        # Get mask for items
        for image in batch:
            batch_mask = np.append(batch_mask, self.masks[image])

        return batch, batch_mask

    def train_val_split(self, percentage=0.1):
        size = len(self.files)


