import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected  # the VSCode says its error but not # TODO: test this code
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

    # Construcción de la estructura de Autoencoder
    input_layer = tf.placeholder(np.float64, shape=[None, n_input])  #  Placeholder es una variable que se le asignará datos.
    hidden = fully_connected(input_layer, n_hidden, activation_fn=None)
    # si activation_fn=None entonces activación lineal
    outputs = fully_connected(hidden, n_output, activation_fn=None)
    reconstruction_loss = tf.reduce_mean(tf.square(outputs-input_layer))
    # Función de costo igual a MSE
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(reconstruction_loss)
    init=tf.global_variables_initializer()

    codings = hidden  # la salida de la capa oculta provee los codings
    hidden_values = None
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            training_op.run(feed_dict = {input_layer: X_train}) # no labels
        codings_val = codings.eval(feed_dict = {input_layer:X_test})
        reconstructed = outputs.eval(feed_dict = {hidden: codings_val})
    return codings_val, reconstructed
