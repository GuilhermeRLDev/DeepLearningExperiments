import numpy as np
import tensorflow as tf


# Image generator responsible for reading images from dataset
class ClothingGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset, validation=False, size=1):
        self.batch_size = batch_size
        self.dataset = dataset
        self.size = size

        if (validation):
            self.files = self.dataset.validation
        else:
            self.files = self.dataset.training

        self.epoch = 0

    def get_size(self):
        return (len(self.files)) // self.batch_size

    def __len__(self):
        return len(self.files) // self.batch_size

    def format_image(self, image, mask):
        image[mask == 0] = 0

        return image

    def __getitem__(self, index):
        # Check if that is higher than actual size
        start = index * self.batch_size
        end = start + self.batch_size
        if (end > len(self.files)):
            start = 0
            end = start + self.batch_size

        images = self.files[start:end]
        masks = []
        images_file = []

        for image in images:
            label = self.dataset.load_image(self.dataset.masks[image], None)
            image = self.dataset.load_image(image, self.dataset.location)
            image = tf.image.per_image_standardization(image)
            image = self.format_image(np.array(image), np.array(label))

            images_file.append(image)
            masks.append(label)

        return np.array(images_file), np.array(masks)

    def on_epoch_end(self):
        self.epoch += 1

    def generate_real_samples(self, n_samples, patch_shape):
        indexes = np.random.randint(0, len(self.files), n_samples)

        samples = self.files[indexes]
        train_a = []
        train_b = []

        for sample in samples:
            label = self.dataset.load_image(self.dataset.masks[sample], None)
            train_b.append(label)
            image = self.dataset.load_image(sample, self.dataset.location)
            image = tf.image.per_image_standardization(image)
            image = self.format_image(np.array(image), np.array(label))
            train_a.append(image)

        y = np.ones((n_samples, patch_shape, patch_shape, 1))

        return [np.array(train_a), np.array(train_b)], y

    def generate_fake_samples(self, g_model, samples, patch_shape):
        X = g_model.predict(samples)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))

        return X, y