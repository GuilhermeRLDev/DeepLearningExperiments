import numpy as np
import matplotlib.pyplot as plt
from models import Pix2PixModel
from src import clothing_generator
from datasets.clothing_dataset_source import Dataset

def train_cloting_model_pix2pix(model, dataset, save_weights):
    generator = clothing_generator.ClothingGenerator(36, dataset)
    model.train(generator, n_ephocs=10)

    if (save_weights):
        model.gan_model.save_weights("/content/drive/My Drive/model_weights/gan_label_to_image")
        model.g_model.save_weights("/content/drive/My Drive/model_weights/g_label_to_image")
        model.d_model.save_weights("/content/drive/My Drive/model_weights/d_label_to_image")


def run_experiments(train=False, load_weights = False):
    dataset = Dataset('resized/segmentation/', 'MASKS')
    validation = clothing_generator.ClothingGenerator(36, dataset, True)
    model = Pix2PixModel.Pix2PixModel((256, 256, 1), (256, 256, 3))

    if train:
        train_cloting_model_pix2pix()

    if load_weights:
        model.gan_model.load_weights("/content/drive/My Drive/model_weights/gan_label_to_image")
        model.g_model.load_weights("/content/drive/My Drive/model_weights/g_label_to_image")
        model.d_model.load_weights("/content/drive/My Drive/model_weights/d_label_to_image")

    images, labels = validation.__getitem__(12)
    print(np.shape(labels))
    prediction = model.g_model.predict(labels)

    prediction = np.reshape(prediction, (36, 256, 256, 3))
    dataset.show_image(labels[5])
    plt.figure()
    dataset.show_image(prediction[5])
    plt.figure()

    dataset.show_image(images[5])





