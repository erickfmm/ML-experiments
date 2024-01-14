# taken from:
# https://medium.com/sigmoid/a-brief-introduction-to-gans-and-how-to-code-them-2620ee465c30
# https://github.com/sarvasvkulpati/intro_to_gans/blob/master/intro_to_gans.ipynb
import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU
from keras.datasets import mnist
from keras.optimizers.legacy import Adam
from keras import initializers
from tensorflow import Variable


def get_optimizer():
    adam = Adam(learning_rate=0.01)
    return adam


def get_generator(optimizer, random_dim, dense_neurons=[256, 512, 1024, 784]):
    generator = Sequential()
    generator.add(Dense(dense_neurons[0], input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(dense_neurons[1]))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(dense_neurons[2]))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(dense_neurons[3], activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator


def get_discriminator(optimizer, input_dim=784, dense_neurons=[1024, 512, 256]):
    discriminator = Sequential()
    discriminator.add(Dense(dense_neurons[0], input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(dense_neurons[1]))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(dense_neurons[2]))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


def plot_generated_images(epoch, generator, random_dim, random_gen, examples=100, dim=(10, 10), figsize=(10, 10), image_size=(28, 28)):
    noise = random_gen.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, image_size[0], image_size[1])

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)


def get_gan_network(discriminator, random_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan


def train(random_dim, random_gen, x_train, epochs=1, batch_size=128):
    # Split the training data into batches of size 128
    batch_count = len(x_train) / batch_size

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam, random_dim)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(int(batch_count))): #tqdm(range(int(batch_count))):
            # Get a random set of input noise and images
            noise = random_gen.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[random_gen.randint(0, len(x_train), size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = random_gen.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)
    return generator, discriminator, gan


from mlexperiments.load_data.loader.downloadable.mnist_keras import LoadMnist
from mlexperiments.utils.keras_persistence.all_inside import save, load
import os
import shutil


class GanMnist:
    def __init__(self, seed=None, random_dim=100):
        # np.random.seed(seed)
        # The dimension of our random noise vector.
        self.random_dim = random_dim
        self.random_gen = np.random.RandomState(seed)
        self.input_dim = 784

    def get_data(self):
        l = LoadMnist()
        return l.get_X_Y()

    def train(self, epochs=1, batch_size=128, to_save=False):
        x, y = self.get_data()
        x = x.reshape(x.shape[0], 784)
        self.generator, self.discriminator, self.gan = train(self.random_dim, self.random_gen, x, epochs, batch_size)
        if to_save:
            self.save()
        return self.generator, self.discriminator, self.gan

    def generate_image(self, noise=None):
        if noise is None:
            noise = self.random_gen.normal(0, 1, size=[1, self.random_dim])
        gen_img = self.generator.predict(noise)
        gen_img = gen_img.reshape(28,28)
        return gen_img

    def save(self):
        try:
            base_folder = "data/created_models"
            if not os.path.exists(base_folder):
                os.mkdir(base_folder)
            newfolder = os.path.join(base_folder, "gan_mnist_v1")
            if not os.path.exists(newfolder):
                os.mkdir(newfolder)
            else:
                shutil.rmtree(newfolder, ignore_errors=True)
                os.mkdir(newfolder)
            save(self.generator, os.path.join(newfolder, "generator.keras_model"))
            save(self.discriminator, os.path.join(newfolder, "discriminator.keras_model"))
            save(self.gan, os.path.join(newfolder, "gan.keras_model"))
        except Exception as e:
            print(f"error in saving: {e}")

    def load(self):
        base_folder = "data/created_models"
        newfolder = os.path.join(base_folder, "gan_mnist_v1")
        try:
            if os.path.exists(newfolder):
                gen_filename = os.path.join(newfolder, "generator.keras_model")
                disc_filename = os.path.join(newfolder, "discriminator.keras_model")
                gan_filename = os.path.join(newfolder, "gan.keras_model")
                if os.path.exists(gen_filename) and os.path.exists(disc_filename) and os.path.exists(gan_filename):
                    self.generator = load(gen_filename)
                    self.discriminator = load(disc_filename)
                    self.gan = load(gan_filename)
                else:
                    print("files doesn't exists")
            else:
                print("folder doesn't exists")
        except Exception as e:
            print("unknown error in loading ", e)



def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap='gray')
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
if __name__ == "__main__":
    g = GanMnist(3335, 10)
    g.train(100, 1024)
    g.save()
    #g.load()
    plot_figures({
        1: g.generate_image(),
        2: g.generate_image(),
        3: g.generate_image(),
        4: g.generate_image(),
        5: g.generate_image(),
        6: g.generate_image(),
        7: g.generate_image(),
        8: g.generate_image(),
        9: g.generate_image(),
        10: g.generate_image(),
        11: g.generate_image(),
        12: g.generate_image(),
        13: g.generate_image(),
        14: g.generate_image(),
        15: g.generate_image(),
        16: g.generate_image(),
        17: g.generate_image(),
        18: g.generate_image(),
        19: g.generate_image(),
        20: g.generate_image(),
        21: g.generate_image(),
        22: g.generate_image(),
        23: g.generate_image(),
        24: g.generate_image(),
        25: g.generate_image(),
        26: g.generate_image(),
        27: g.generate_image(),
        28: g.generate_image(),
        29: g.generate_image(),
        30: g.generate_image()
    },5,6)
    plt.show()
