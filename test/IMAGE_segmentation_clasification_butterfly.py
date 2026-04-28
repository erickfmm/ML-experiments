import os
import sys


from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

from mlexperiments.load_data.loader.image.butterfly_segment import LoadButterflySegment
import mlexperiments.utils.image.resize_image as resize
import mlexperiments.preprocessing.image2D.rgb_ypbpr as ypbpr
import mlexperiments.preprocessing.image2D.convolution as conv
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sn
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
#  Hyperparameters from environment (set by the web GUI)
# ═══════════════════════════════════════════════════════════════════

def get_env_int(key, default):
    return int(os.environ.get(key, default))

def get_env_float(key, default):
    return float(os.environ.get(key, default))


def save_as_pickle(new_size:int = 100, to_gray: bool = True, to_pickle=True):
    lb = LoadButterflySegment()
    lb.download()
    # new_size = 100
    x_gray = []
    x_seg = []
    y = []
    i = 0
    for im, seg, btype in lb.get_all_yielded():
        print(i)
        i = i+1
        im2 = resize.with_image_module(im, new_size)
        if to_gray:
            im2 = im2.convert("L")  # to gray
        im2 = np.asarray(im2)
        x_gray.append(im2)
        seg2 = resize.with_image_module(seg, new_size)
        if to_gray:
            seg2 = seg2.convert("L")
        x_seg.append(np.asarray(seg2))
        y.append(btype)
    if to_pickle:
        with open("data/created_models/butterfly.pkl", "wb") as file_handle:
            print("writing pickle xgray, type")
            pickle.dump({"x": x_gray,"y": y}, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("data/created_models/butterfly_segment.pkl", "wb") as file_handle:
            print("writing pickle xgray, xseg")
            pickle.dump({"x": x_gray,"y": x_seg}, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return x_gray, x_seg, y


def open_pickle(which_pkl:str) -> dict:
    """
    Parameters
    ----------
    which_pkl:str: 'butterfly' to open the pickle with "y" = types of butterfly or
    'segment' to open the pickle file with "y" = the segmentation of file

    Returns
    -------
    Dict: A Dictionary with two keys: "x" (an array of images) 
    and "y" (an array of int if 'butterfly' or images if 'segment')
    """
    path_to_pickle = "data/created_models/butterfly_segment.pkl" \
        if which_pkl == "segment" else "data/created_models/butterfly.pkl"
    with open(path_to_pickle, "rb") as file_handler:
        return pickle.load(file_handler)


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # GlobalMaxPooling2D equivalent
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_classifier(x, y, epochs=100, batch_size=16, learning_rate=0.001):
    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Print model summary
    print(model)
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    x_t = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/total:.4f} - accuracy: {correct/total:.4f}")
    return model

def simple_classifier_pipeline(x, y, epochs=100, batch_size=16, learning_rate=0.001, test_split=0.33):
    print("cantidad de etiquetas: ", len(set(y)))
    y_encoded = pd.get_dummies(y).astype('float32').values
    y_labels = np.argmax(y_encoded, axis=1)
    x = np.asarray(x)
    print("x shape: ", x.shape)
    print("y shape: ", y_labels.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y_labels, test_size=test_split, random_state=1992)
    model = train_classifier(x_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    print(model)

    # Evaluate
    model.eval()
    x_test_t = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    with torch.no_grad():
        outputs = model(x_test_t)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()

    # Save model
    os.makedirs("data/created_models", exist_ok=True)
    torch.save(model.state_dict(), "data/created_models/butterfly_classifier.pt")
    print("✅ Classifier model saved to data/created_models/butterfly_classifier.pt")

    confusion = confusion_matrix(y_test, y_pred, normalize='pred')
    print(confusion)
    df_cm = pd.DataFrame(confusion, range(10), range(10))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    plt.show()

class SimpleSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(3, 3)
        self.dropout = nn.Dropout(0.1)

        # Bottleneck
        self.bottleneck = nn.Conv2d(32, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Decoder
        self.up3 = nn.Upsample(scale_factor=3, mode='nearest')
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=5, padding=0),
            nn.ReLU(),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 32, 3, kernel_size=3, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        x1 = self.dropout(self.enc1(x))       # (N, 32, 100, 100)
        x = self.pool(x1)                      # (N, 32, 50, 50)
        x2 = self.dropout(self.enc2(x))        # (N, 32, 50, 50)
        x = self.pool(x2)                      # (N, 32, 25, 25)
        x3 = self.dropout(self.enc3(x))        # (N, 32, 25, 25)
        x = self.pool3(x3)                     # (N, 32, 8, 8)

        # Bottleneck
        x = self.relu(self.bottleneck(x))      # (N, 256, 8, 8)

        # Decoder
        x = self.up3(x)                        # (N, 256, 24, 24)
        x3_crop = x3[:, :, 1:25, 1:25]         # crop to match
        x = torch.cat([x, x3_crop], dim=1)     # (N, 288, 24, 24)
        x = self.dropout(self.dec3(x))         # (N, 32, 22, 22)

        x = self.up2(x)                        # (N, 32, 44, 44)
        x2_crop = x2[:, :, 3:47, 3:47]
        x = torch.cat([x, x2_crop], dim=1)
        x = self.dropout(self.dec2(x))

        x = self.up1(x)
        # Pad to 100x100
        x = nn.functional.pad(x, (4, 4, 4, 4))  # pad to 100x100
        x1_crop = x1[:, :, :x.size(2), :x.size(3)]
        x = torch.cat([x, x1_crop], dim=1)
        x = self.dec1(x)
        # Ensure output is 100x100
        x = nn.functional.interpolate(x, size=(100, 100), mode='bilinear', align_corners=False)
        return x


def train_segmentation(x_input, y_input, epochs=100, batch_size=16, learning_rate=0.001):
    model = SimpleSegmentation()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    x_t = torch.tensor(x_input, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    y_t = torch.tensor(y_input, dtype=torch.float32).permute(0, 3, 1, 2)
    dataset = TensorDataset(x_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(model)
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/len(x_t):.4f}")
    return model


def show_segmentation_interface(model, x_test, y_test):
    model.eval()
    x_test_t = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    with torch.no_grad():
        pred = model(x_test_t).permute(0, 2, 3, 1).numpy()  # back to (N, H, W, C)
    # Initialize the index variable
    global idx
    idx = 0
    import tkinter as tk
    from tkinter import filedialog
    from PIL import Image, ImageTk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    # Create a function to update the displayed images
    def update_images():
        ax1.clear()
        ax1.imshow(pred[idx])
        ax1.set_title('Predicted Image')
        ax1.axis('off')
        ax2.clear()
        ax2.imshow(y_test[idx])
        ax2.set_title('Actual Image')
        ax2.axis('off')
        ax3.clear()
        ax3.imshow(x_test[idx])
        ax3.set_title('Full Image')
        ax3.axis('off')
        ax4.clear()
        ax4.imshow(x_test[idx]*pred[idx])
        ax4.set_title('Cropped Image')
        ax4.axis('off')
        canvas.draw()

    # Function to go to the previous image
    def prev_image():
        global idx
        if idx > 0:
            idx -= 1
            update_images()

    # Function to go to the next image
    def next_image():
        global idx
        if idx < len(pred) - 1:
            idx += 1
            update_images()

    # Function to upload an image and make a prediction
    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ppm *.pgm")])
        if file_path:
            image = Image.open(file_path)
            image = image.resize((100, 100))  # Resize the image to match model input size
            image = np.array(image) / 255.0  # Normalize the image
            with torch.no_grad():
                img_tensor = torch.tensor(np.expand_dims(image, axis=0), dtype=torch.float32).permute(0, 3, 1, 2)
                prediction = model(img_tensor).permute(0, 2, 3, 1).numpy()[0]
            idx = 0  # Reset the index
            ax3.clear()
            ax3.imshow(image)
            ax3.set_title('Uploaded Image')
            ax3.axis('off')

            ax1.clear()
            ax1.imshow(prediction)
            ax1.set_title('Predicted Segmentation')
            ax1.axis('off')

            ax2.clear()
            ax2.axis('off')

            ax4.clear()
            ax4.imshow(prediction*image)
            ax4.set_title('Cropped Segmentation')
            ax4.axis('off')
            
            canvas.draw()
    # Create the main application window
    root = tk.Tk()
    root.title("Image Viewer")

    # Create Matplotlib figure
    fig = Figure(figsize=(8, 4), dpi=100)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)

    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Create a Tkinter canvas for the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # Create navigation buttons
    prev_button = tk.Button(root, text="Previous", command=prev_image)
    next_button = tk.Button(root, text="Next", command=next_image)
    upload_button = tk.Button(root, text="Upload Image", command=upload_image)
    prev_button.pack()
    next_button.pack()
    upload_button.pack()

    # Display the initial image
    update_images()

    # Start the Tkinter main loop
    root.mainloop()

def segmentation_pipeline(x, x_seg, epochs=100, batch_size=16, learning_rate=0.001, test_split=0.33):
    x_seg = np.asarray(x_seg)
    x_seg = np.nan_to_num(x_seg)
    x = np.asarray(x)
    x = np.nan_to_num(x)
    print("len x ",len(x))
    print("shape x", x.shape)
    print("shape x_seg", x_seg.shape)
    x = x/255.0
    x_seg = x_seg/255.0
    x_train, x_test, y_train, y_test = train_test_split(x, x_seg, test_size=test_split, random_state=1992)
    print("x_train shape", x_train.shape)
    print("y_train shape", y_train.shape)
    print("x_test shape", x_test.shape)
    print("y_test shape", y_test.shape)
    filepath = "data/created_models/butterfly_segmentation.pt"
    if os.path.exists(filepath):
        model = SimpleSegmentation()
        model.load_state_dict(torch.load(filepath, weights_only=True))
        model.eval()
    else:
        model = train_segmentation(x_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        os.makedirs("data/created_models", exist_ok=True)
        torch.save(model.state_dict(), filepath)
    print("✅ Segmentation model saved to", filepath)
    show_segmentation_interface(model, x_test, y_test)


def run_download():
    """Download the butterfly dataset only."""
    print("Downloading butterfly dataset...")
    lb = LoadButterflySegment()
    lb.download()
    print("✅ Download complete.")


def run_save_pickle():
    """Generate pickle files from downloaded dataset."""
    image_size = get_env_int("BUTTERFLY_IMAGE_SIZE", 100)
    print(f"Generating pickle files (image size: {image_size}px)...")
    save_as_pickle(new_size=image_size, to_gray=False, to_pickle=True)
    print("✅ Pickle files generated.")


def run_train_segmentation():
    """Train segmentation model with hyperparameters from env."""
    epochs = get_env_int("BUTTERFLY_EPOCHS", 100)
    batch_size = get_env_int("BUTTERFLY_BATCH_SIZE", 16)
    learning_rate = get_env_float("BUTTERFLY_LR", 0.001)
    test_split = get_env_float("BUTTERFLY_TEST_SPLIT", 0.33)

    print(f"Loading segmentation data...")
    data = open_pickle("segment")
    x = np.asarray(data["x"])
    x_seg = np.asarray(data["y"])
    x = np.nan_to_num(x)
    x_seg = np.nan_to_num(x_seg)
    print("len x ", len(x))
    print("shape x", x.shape)
    print("shape x_seg", x_seg.shape)
    x = x / 255.0
    x_seg = x_seg / 255.0
    x_train, x_test, y_train, y_test = train_test_split(x, x_seg, test_size=test_split, random_state=1992)
    print("x_train shape", x_train.shape)
    print("y_train shape", y_train.shape)

    model = train_segmentation(x_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    os.makedirs("data/created_models", exist_ok=True)
    filepath = "data/created_models/butterfly_segmentation.pt"
    torch.save(model.state_dict(), filepath)
    print(f"✅ Segmentation model saved to {filepath}")


def run_train_classifier():
    """Train classifier model with hyperparameters from env."""
    epochs = get_env_int("BUTTERFLY_EPOCHS", 100)
    batch_size = get_env_int("BUTTERFLY_BATCH_SIZE", 16)
    learning_rate = get_env_float("BUTTERFLY_LR", 0.001)
    test_split = get_env_float("BUTTERFLY_TEST_SPLIT", 0.33)

    print(f"Loading classifier data...")
    data = open_pickle("butterfly")
    x = np.asarray(data["x"])
    y = data["y"]

    # If grayscale, expand to 3 channels for the classifier
    if len(x.shape) == 3:
        x = np.stack([x, x, x], axis=-1)
    x = x / 255.0 if x.max() > 1.0 else x

    print("cantidad de etiquetas: ", len(set(y)))
    y_encoded = pd.get_dummies(y).astype('float32').values
    y_labels = np.argmax(y_encoded, axis=1)
    print("x shape: ", x.shape)
    print("y shape: ", y_labels.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y_labels, test_size=test_split, random_state=1992)
    model = train_classifier(x_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    # Evaluate
    model.eval()
    x_test_t = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    with torch.no_grad():
        outputs = model(x_test_t)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()

    # Save model
    os.makedirs("data/created_models", exist_ok=True)
    torch.save(model.state_dict(), "data/created_models/butterfly_classifier.pt")
    print("✅ Classifier model saved to data/created_models/butterfly_classifier.pt")

    confusion = confusion_matrix(y_test, y_pred, normalize='pred')
    print("Confusion matrix:")
    print(confusion)


if __name__ == "__main__":
    task = os.environ.get("BUTTERFLY_TASK", "")

    if task == "download":
        run_download()
    elif task == "save_pickle":
        run_save_pickle()
    elif task == "train_segmentation":
        run_train_segmentation()
    elif task == "train_classifier":
        run_train_classifier()
    else:
        # Original behavior: run full pipeline interactively
        print("running as main (full pipeline)")
        x, x_seg, y = save_as_pickle(new_size=100, to_gray=False, to_pickle=False)
        #simple_classifier_pipeline(x, y)
        segmentation_pipeline(x, x_seg)