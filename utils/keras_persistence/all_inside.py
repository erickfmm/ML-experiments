
from keras.models import load_model

def save(model, filename: str):
    # save model and architecture to single file
    model.save(filename)
    print("Saved model to disk")
    return model

def load(filename: str):
    # load model
    model = load_model(filename)
    # summarize model.
    model.summary()
    return model