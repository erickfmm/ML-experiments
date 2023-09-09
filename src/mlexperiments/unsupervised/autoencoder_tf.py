from os.path import join
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# from sklearn import datasets
# from sklearn.model_selection import train_test_split


# iris = datasets.load_iris()
# dataX = iris.data
# datay = iris.target
# X_train, X_test, y_train, y_test = train_test_split(dataX, datay, test_size = 0.3, random_state=0)


def reconstruct_data(X_train, X_test, n_hidden=2, learning_rate=0.1, n_iterations=1000):
    if len(X_train) < 1:
        raise ValueError("X has no length")
    n_input = len(X_train[0])
    n_output = n_input
    # TODO: Reimplement using keras
    input = keras.Input(shape=(n_input, ))
    hidden = layers.Dense(n_hidden, activation='relu', name="hidden")
    x = hidden(input)
    output = layers.Dense(n_input, activation='relu')(x)
    model = keras.Model(inputs=input, outputs=output)
    keras.utils.plot_model(model, join("created_models", "reconstructed_model.png"), show_shapes=True)

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.SGD(),
        metrics=["mse"],
    )
    model.fit(X_train, X_train, batch_size=2, epochs=3, validation_split=0.2)
    model.save(join("created_models", "reconstructed.model"))
    reconstructed = model.predict(X_test)
    

    intermediate_layer_model = keras.Model(inputs=model.input,
                                 outputs=model.get_layer("hidden").output)
    codings_val = intermediate_layer_model.predict(X_test)
    
    return codings_val, reconstructed
