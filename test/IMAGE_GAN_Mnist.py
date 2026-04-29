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

import torch
import torch.nn as nn
import torch.optim as optim


OUTPUT_DIR = os.path.join("data", "created_models", "gan")


def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid integer for {name}: {value!r}. Using default {default}.")
        return default


def _get_env_bool(name, default=False):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _prepare_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


class Generator(nn.Module):
    def __init__(self, random_dim, dense_neurons=None):
        super().__init__()
        if dense_neurons is None:
            dense_neurons = [256, 512, 1024, 784]
        self.model = nn.Sequential(
            nn.Linear(random_dim, dense_neurons[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(dense_neurons[0], dense_neurons[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(dense_neurons[1], dense_neurons[2]),
            nn.LeakyReLU(0.2),
            nn.Linear(dense_neurons[2], dense_neurons[3]),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=784, dense_neurons=None):
        super().__init__()
        if dense_neurons is None:
            dense_neurons = [1024, 512, 256]
        self.model = nn.Sequential(
            nn.Linear(input_dim, dense_neurons[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(dense_neurons[0], dense_neurons[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(dense_neurons[1], dense_neurons[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(dense_neurons[2], 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def plot_generated_images(epoch, generator, random_dim, random_gen, examples=100, dim=(10, 10), figsize=(10, 10), image_size=(28, 28)):
    generator.eval()
    noise = torch.tensor(random_gen.normal(0, 1, size=[examples, random_dim]).astype(np.float32))
    with torch.no_grad():
        generated_images = generator(noise).numpy()
    generated_images = generated_images.reshape(examples, image_size[0], image_size[1])

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    _prepare_output_dir()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_generated_image_epoch_%d.png' % epoch))


def train(random_dim, random_gen, x_train, epochs=1, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build our GAN network
    generator = Generator(random_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=0.0002)
    opt_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    x_train_t = torch.tensor(x_train.astype(np.float32), device=device)
    dataset = torch.utils.data.TensorDataset(x_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for (image_batch,) in tqdm(loader):
            current_batch_size = image_batch.size(0)

            # Generate fake images
            noise = torch.randn(current_batch_size, random_dim, device=device)
            with torch.no_grad():
                generated_images = generator(noise)
            X = torch.cat([image_batch, generated_images])
            y_dis = torch.zeros(2 * current_batch_size, device=device)
            y_dis[:current_batch_size] = 0.9  # One-sided label smoothing

            # Train discriminator
            discriminator.train()
            opt_d.zero_grad()
            d_output = discriminator(X).squeeze()
            d_loss = criterion(d_output, y_dis)
            d_loss.backward()
            opt_d.step()

            # Train generator
            noise = torch.randn(current_batch_size, random_dim, device=device)
            y_gen = torch.ones(current_batch_size, device=device)

            opt_g.zero_grad()
            g_output = generator(noise)
            d_fake = discriminator(g_output).squeeze()
            g_loss = criterion(d_fake, y_gen)
            g_loss.backward()
            opt_g.step()

    return generator, discriminator


from mlexperiments.load_data.loader.downloadable.mnist_torch import LoadMnist
import shutil


class GanMnist:
    def __init__(self, seed=None, random_dim=100):
        self.random_dim = random_dim
        self.random_gen = np.random.RandomState(seed)
        self.input_dim = 784
        self.generator = None
        self.discriminator = None

    def get_data(self):
        l = LoadMnist()
        return l.get_X_Y()

    def train(self, epochs=1, batch_size=128, to_save=False):
        x, y = self.get_data()
        x = x.reshape(x.shape[0], 784).astype(np.float32)
        # Normalize to [-1, 1] for Tanh
        x = (x / 255.0) * 2 - 1
        self.generator, self.discriminator = train(self.random_dim, self.random_gen, x, epochs, batch_size)
        if to_save:
            self.save()
        return self.generator, self.discriminator

    def generate_image(self, noise=None):
        device = next(self.generator.parameters()).device
        if noise is None:
            noise = torch.tensor(self.random_gen.normal(0, 1, size=[1, self.random_dim]).astype(np.float32)).to(device)
        self.generator.eval()
        with torch.no_grad():
            gen_img = self.generator(noise)
        gen_img = gen_img.cpu().numpy().reshape(28, 28)
        return gen_img

    def save(self):
        try:
            base_folder = "data/created_models"
            os.makedirs(base_folder, exist_ok=True)

            newfolder = os.path.join(base_folder, "gan_mnist_v1")
            if not os.path.exists(newfolder):
                os.mkdir(newfolder)
            else:
                shutil.rmtree(newfolder, ignore_errors=True)
                os.mkdir(newfolder)

            generator_checkpoint = {
                "state_dict": self.generator.state_dict(),
                "random_dim": self.random_dim,
            }
            discriminator_checkpoint = {
                "state_dict": self.discriminator.state_dict(),
            }

            # Legacy paths kept for backwards compatibility.
            torch.save(self.generator.state_dict(), os.path.join(newfolder, "generator.pt"))
            torch.save(self.discriminator.state_dict(), os.path.join(newfolder, "discriminator.pt"))

            # Flat files used by the web GUI endpoints.
            torch.save(generator_checkpoint, os.path.join(base_folder, "gan_generator.pth"))
            torch.save(discriminator_checkpoint, os.path.join(base_folder, "gan_discriminator.pth"))
        except Exception as e:
            print(f"error in saving: {e}")

    def load(self):
        base_folder = "data/created_models"
        newfolder = os.path.join(base_folder, "gan_mnist_v1")
        try:
            gen_candidates = [
                os.path.join(base_folder, "gan_generator.pth"),
                os.path.join(base_folder, "gan_generator.pt"),
                os.path.join(newfolder, "generator.pt"),
            ]
            disc_candidates = [
                os.path.join(base_folder, "gan_discriminator.pth"),
                os.path.join(base_folder, "gan_discriminator.pt"),
                os.path.join(newfolder, "discriminator.pt"),
            ]

            gen_filename = next((path for path in gen_candidates if os.path.exists(path)), None)
            disc_filename = next((path for path in disc_candidates if os.path.exists(path)), None)

            if not gen_filename or not disc_filename:
                print("GAN model files don't exist")
                return

            gen_checkpoint = torch.load(gen_filename, map_location="cpu", weights_only=False)
            if isinstance(gen_checkpoint, dict) and "state_dict" in gen_checkpoint:
                gen_state_dict = gen_checkpoint["state_dict"]
                random_dim = int(gen_checkpoint.get("random_dim", self.random_dim))
            else:
                gen_state_dict = gen_checkpoint
                random_dim = gen_state_dict["model.0.weight"].shape[1]

            disc_checkpoint = torch.load(disc_filename, map_location="cpu", weights_only=False)
            disc_state_dict = disc_checkpoint["state_dict"] if isinstance(disc_checkpoint, dict) and "state_dict" in disc_checkpoint else disc_checkpoint

            self.random_dim = random_dim
            self.generator = Generator(self.random_dim)
            self.generator.load_state_dict(gen_state_dict)
            self.generator.eval()
            self.discriminator = Discriminator()
            self.discriminator.load_state_dict(disc_state_dict)
            self.discriminator.eval()
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
    _prepare_output_dir()
    preview_path = os.path.join(OUTPUT_DIR, "latest_preview.png")
    plt.savefig(preview_path, dpi=160, bbox_inches='tight')
    print(f"Saved plot: {preview_path}")
if __name__ == "__main__":
    seed = _get_env_int("ML_SEED", 3335)
    random_dim = _get_env_int("ML_RANDOM_DIM", 100)
    epochs = _get_env_int("ML_EPOCHS", 100)
    batch_size = _get_env_int("ML_BATCH_SIZE", 128)
    running_from_web = any(
        key in os.environ for key in ("ML_SEED", "ML_RANDOM_DIM", "ML_EPOCHS", "ML_BATCH_SIZE")
    )
    show_plot = _get_env_bool("ML_SHOW_PLOT", default=(not running_from_web and bool(os.getenv("DISPLAY"))))

    print(
        f"Starting GAN training with seed={seed}, random_dim={random_dim}, "
        f"epochs={epochs}, batch_size={batch_size}"
    )

    g = GanMnist(seed, random_dim)
    g.train(epochs, batch_size)
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
    if show_plot:
        plt.show()
    else:
        plt.close('all')
