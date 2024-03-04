from os.path import join, exists
from os import mkdir
from tensorflow import keras
from keras import layers

def MLP_reconstruct_data(X_train, X_test, n_hidden=2, epochs=10, name="reconstructed_model"):
    if len(X_train) < 1:
        raise ValueError("X has no length")
    n_input = len(X_train[0])
    n_output = n_input
    # TODO: Reimplement using keras
    input = keras.Input(shape=(n_input, ))
    hidden = layers.Dense(n_hidden, activation='sigmoid', name="hidden")
    x = hidden(input)
    output = layers.Dense(n_input, activation='sigmoid')(x)
    model = keras.Model(inputs=input, outputs=output)
    if not exists(join("data/created_models",name)):
        mkdir(join("data/created_models",name))
    #keras.utils.plot_model(model, join("data/created_models", name, "reconstructed_model.png"), show_shapes=True)

    model.compile(
        loss="mean_squared_error",
        optimizer='adam',
        metrics=["accuracy"],
    )
    model.fit(X_train, X_train, batch_size=128, epochs=epochs, validation_split=0.2)
    #model.save("data/created_models/"+name+"/reconstructed.model")
    reconstructed = model.predict(X_test)
    

    intermediate_layer_model = keras.Model(inputs=model.input,
                                 outputs=model.get_layer("hidden").output)
    codings_val = intermediate_layer_model.predict(X_test)
    
    return codings_val, reconstructed
