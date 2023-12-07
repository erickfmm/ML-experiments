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

import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sn
import matplotlib.pyplot as plt

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


def simple_classifier(x, y):
    model = keras.Sequential()
    model.add(keras.Input(shape=(100,100,3)))
    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D(3))
    

    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary() #only for printing purposes
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )
    model.fit(x, y, epochs=100, batch_size=16)
    return model

def simple_classifier_pipeline(x, y):
    print("cantidad de etiquetas: ", len(set(y)))
    #y = keras.utils.to_categorical(y)
    y = pd.get_dummies(y).astype('float32').values
    x = np.asarray(x)
    y = np.asarray(y)
    print(y[0])
    print("x shape: ", np.asarray(x).shape)
    print("y shape: ", np.asarray(y).shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1992)
    model = simple_classifier(x_train, y_train)
    model.summary()
    model.save("data/created_models/butterfly_classifier.keras")
    model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred=np.argmax(y_pred, axis=1)
    y_test=np.argmax(y_test, axis=1)
    confusion = confusion_matrix(y_test, y_pred, normalize='pred')
    print(confusion)
    df_cm = pd.DataFrame(confusion, range(10), range(10))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    plt.show()

def simple_segmentation(x_input, y_input):
    inputs = layers.Input(shape=(100,100,3))
    x1 = layers.Conv2D(32, 5, activation="relu", padding="same")(inputs)
    x = layers.Dropout(0.1)(x1)
    x = layers.MaxPooling2D((2,2), padding="same")(x)
    x2 = layers.Conv2D(32, 5, activation="relu", padding="same")(x)
    x = layers.Dropout(0.1)(x2)
    x = layers.MaxPooling2D((2,2), padding="same")(x)
    x3 = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.Dropout(0.1)(x3)
    x = layers.MaxPooling2D((3,3), padding="same")(x)

    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)

    x = layers.UpSampling2D(size=(3,3))(x)
    x3 = layers.ZeroPadding2D(padding=(1, 1))(x3)
    x = layers.concatenate([x, x3], axis=-1)
    x = layers.Conv2D(32, 3, activation="relu", padding="valid")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.UpSampling2D(size=(2,2))(x)
    #x2 = layers.ZeroPadding2D(2)(x2)
    x = layers.concatenate([x, x2], axis=-1)
    x = layers.Conv2D(32, 5, activation="relu", padding="valid")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.ZeroPadding2D(padding=(4, 4))(x)
    x = layers.concatenate([x, x1], axis=-1)
    output = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs=[inputs], outputs=[output])
    model.summary() #only for printing purposes
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
        )
    model.fit(x_input, y_input, epochs=100, batch_size=16)
    return model


def show_segmentation_interface(model, x_test, y_test):
    pred = model.predict(x_test)
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
            prediction = model.predict(np.expand_dims(image, axis=0))[0]
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

def segmentation_pipeline(x, x_seg):
    x_seg = np.asarray(x_seg)
    x_seg = np.nan_to_num(x_seg)
    x = np.asarray(x)
    x = np.nan_to_num(x)
    print("len x ",len(x))
    print("shape x", x.shape)
    print("shape x_seg", x_seg.shape)
    x = x/255.0
    x_seg = x_seg/255.0
    x_train, x_test, y_train, y_test = train_test_split(x, x_seg, test_size=0.33, random_state=1992)
    print("x_train shape", x_train.shape)
    print("y_train shape", y_train.shape)
    print("x_test shape", x_test.shape)
    print("y_test shape", y_test.shape)
    filepath = "data/created_models/butterfly_segmentation.keras"
    if os.path.exists(filepath):
        model = keras.saving.load_model(filepath)
    else:
        model = simple_segmentation(x_train, y_train)
        model.save(filepath)
    show_segmentation_interface(model, x_test, y_test)

if __name__ == "__main__":
    print("running as main")
    x, x_seg, y = save_as_pickle(new_size=100, to_gray=False, to_pickle=False)
    #simple_classifier_pipeline(x, y)
    segmentation_pipeline(x, x_seg)