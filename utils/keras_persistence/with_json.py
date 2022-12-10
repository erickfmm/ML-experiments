from keras.models import model_from_json


def save(model, model_filename: str, weights_filename: str):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_filename)
    print("Saved model to disk")
    return model


def load(model_filename: str, weights_filename: str):
    # load json and create model
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_filename)
    print("Loaded model from disk")
    print("Important!!!! compile model before using")
    return loaded_model
