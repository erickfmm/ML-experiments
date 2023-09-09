# asumes:
# sudo pip install PyYAML
from keras.models import model_from_yaml


def save(model, model_filename: str, weights_filename: str):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(model_filename, "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(weights_filename)
    print("Saved model to disk")
    return model


def load(model_filename: str, weights_filename: str):
    # load YAML and create model
    yaml_file = open(model_filename, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(weights_filename)
    print("Loaded model from disk")
    return loaded_model
