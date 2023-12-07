import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################
 
from mlexperiments.load_data.loader.text_sentiment.ENG_1million_tweets import Load1MTweets
from mlexperiments.load_data.loader.text_sentiment.ENG_imdb_sentiment import LoadImdbSentiment
from mlexperiments.load_data.loader.text_sentiment.ENG_RedditTwitterSentiment import LoadRedditOrTwitterSentiment
from mlexperiments.load_data.loader.text_sentiment.ENG_Sentiment140 import LoadSentiment140
from mlexperiments.load_data.loader.text_sentiment.ENG_Steam import LoadSteam

def get_data():
    print("1M tweets")
    l1 = Load1MTweets(lang="en")
    l1.download()
    x1,y1 = l1.get_X_Y()
    print("setY1:", set(y1))
    print("setMetadata:", set(l1.Metadata))

    print("imdb")
    l2 = LoadImdbSentiment()
    l2.download()
    x2,y2 = l2.get_X_Y()
    print("setY2:", set(y2))

    print("reddit")
    l3 = LoadRedditOrTwitterSentiment(source="reddit")
    l3.download()
    x3,y3 = l3.get_X_Y()
    print("setY3:", set(y3))

    print("twitter")
    l4 = LoadRedditOrTwitterSentiment(source="twitter")
    x4,y4 = l4.get_X_Y()
    print("setY4:", set(y4))

    print("sentiment 140")
    l5 = LoadSentiment140()
    l5.download()
    x5,y5 = l5.get_X_Y()
    print("setY5:", set(y5))

    print("steam")
    l6 = LoadSteam()
    l6.download()
    x6,y6 = l6.get_X_Y()
    print("setY6:", set(y6))

    X = x1
    X.extend(x2)
    X.extend(x3)
    X.extend(x4)
    X.extend(x5)
    X.extend(x6)

    del x1,x2,x3,x4,x5,x6

    Y = y1
    Y.extend(y2)
    Y.extend(y3)
    Y.extend(y4)
    Y.extend(y5)
    Y.extend(y6)
    del y1,y2,y3,y4,y5,y6

    del l1,l2,l3,l4,l5,l6

    print("lenX: ", len(X))
    print("lenY:", len(Y))
    print("lensetY:", len(set(Y)))
    print("setY:", set(Y))
    print("0  :", sum([1 if x==0 else 0 for x in Y]))
    print("0.5:", sum([1 if x==0.5 else 0 for x in Y]))
    print("1  :", sum([1 if x==1 else 0 for x in Y]))
    return X, Y


def downsample(X, Y, ratio):
    print("to downsample")
    n = int(min([sum([1 if x==0 else 0 for x in Y]), 
            sum([1 if x==0.5 else 0 for x in Y]), 
            sum([1 if x==1 else 0 for x in Y])]) * ratio)

    print("n  :", n)
    combined_list0 = [(X[i],Y[i]) for i in range(len(X)) if Y[i] == 0 ]
    combined_list05 = [(X[i],Y[i]) for i in range(len(X)) if Y[i] == 0.5 ]
    combined_list1 = [(X[i],Y[i]) for i in range(len(X)) if Y[i] == 1 ]
    import random
    sampled_combined0 = random.sample(combined_list0, n)
    sampled_combined05 = random.sample(combined_list05, n)
    sampled_combined1 = random.sample(combined_list1, n)

    sampled_combined = sampled_combined0
    sampled_combined.extend(sampled_combined05)
    sampled_combined.extend(sampled_combined1)
    del combined_list0,combined_list05,combined_list1,sampled_combined0,sampled_combined05,sampled_combined1

    X = [x[0] for x in sampled_combined]
    Y = [y[1] for y in sampled_combined]

    print("lenX: ", len(X))
    print("lenY:", len(Y))
    print("lensetY:", len(set(Y)))
    print("setY:", set(Y))
    print("0  :", sum([1 if x==0 else 0 for x in Y]))
    print("0.5:", sum([1 if x==0.5 else 0 for x in Y]))
    print("1  :", sum([1 if x==1 else 0 for x in Y]))
    print("finished downsampling")
    return X, Y

def one_hot_encode(Y):
    print("to one hot encode")
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    idx0 = Y.index(0)
    idx05 = Y.index(0.5)
    idx1 = Y.index(1)
    ### integer mapping using LabelEncoder
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    print(integer_encoded)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    ### One hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    Y = onehot_encoder.fit_transform(integer_encoded)
    del integer_encoded
    print(Y[0])
    mappings = {
        "negative": Y[idx0],
        "neutral": Y[idx05],
        "positive": Y[idx1],
    }
    return Y, mappings


def clean_data(X):
    from mlexperiments.preprocessing.text.clean import clean_tweet
    X = [clean_tweet(x) for x in X]
    return X

def tokenize_with_gensim_spacy(X):
    from gensim.corpora import Dictionary
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", 
                                                 "morphologizer", 
                                                 #"parser", 
                                                 "senter", 
                                                 "attribute_ruler", 
                                                 "lemmatizer", 
                                                 "ner"
                                                 ])
    print("tokenizing")
    X_tokenized = []
    i_doc = 0
    for docstring in X:
        doc_tokenizado = [token.text for token in nlp(docstring)]
        X_tokenized.append(doc_tokenizado)
        i_doc += 1
        if i_doc % 10000 == 0:
            print(i_doc)
    print("to create dictionary")
    dictionary = Dictionary.from_documents(X_tokenized)
    
    print("doc to BoW")
    print(X_tokenized[0])
    X_idxs = [dictionary.doc2idx(doc) for doc in X_tokenized]
    return X_idxs, dictionary, nlp

def tokenize_tfds(X):
    import tensorflow_datasets as tfds
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        X, target_vocab_size=2**16
    )
    X = [tokenizer.encode(sentence) for sentence in X]
    return X, tokenizer

def padding(X, max_len :int = None):
    from tensorflow import keras
    from keras.preprocessing.sequence import pad_sequences
    if max_len is None:
        max_len = max([x for doc in X for x in doc])+1
        #max_len = max([len(sentence) for sentence in X])
    X = pad_sequences(X, maxlen=max_len, padding="post", value=0)
    return X

def idx_to_bow(X_idxs):
    import numpy as np
    new_X = []
    for doc in X_idxs:
        new_doc = []
        for token_idx in doc:
            l = np.zeros(len(doc))
            l[token_idx] = 1
            new_doc.append(l)
        new_X.append(new_doc)
    X = np.asarray(new_X)
    return X

def create_run_copied_model(X, Y, vocab_size):
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1992)
    import tensorflow as tf
    from tensorflow import keras
    from keras import Model, layers
    class DCNN(Model):

        def __init__(self,
                    vocab_size,
                    emb_dim=128,
                    nb_filters=50,
                    FFN_units=512,
                    nb_classes=2,
                    dropout_rate=0.1,
                    training=False,
                    name="dcnn"):
            super(DCNN, self).__init__(name=name)

            self.embedding = layers.Embedding(vocab_size,
                                            emb_dim)
            self.bigram = layers.Conv1D(filters=nb_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
            self.trigram = layers.Conv1D(filters=nb_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
            self.fourgram = layers.Conv1D(filters=nb_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
            self.pool = layers.GlobalMaxPool1D() # No tenemos variable de entrenamiento
                                                # así que podemos usar la misma capa
                                                # para cada paso de pooling <- No
            self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
            self.dropout = layers.Dropout(rate=dropout_rate)
            if nb_classes == 1:
                self.last_dense = layers.Dense(units=1,
                                            activation="sigmoid")
            elif nb_classes == 2:
                self.last_dense = layers.Dense(units=1,
                                            activation="sigmoid")
            else:
                self.last_dense = layers.Dense(units=nb_classes,
                                            activation="softmax")

        def call(self, inputs, training):
            x = self.embedding(inputs)
            x_1 = self.bigram(x)
            x_1 = self.pool(x_1)
            x_2 = self.trigram(x)
            x_2 = self.pool(x_2)
            x_3 = self.fourgram(x)
            x_3 = self.pool(x_3)

            #merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
            merged = layers.concatenate([x_1, x_2, x_3], axis=-1)
            merged = self.dense_1(merged)
            merged = self.dropout(merged, training)
            output = self.last_dense(merged)

            return output
        def get_model(self, X):
            inputs = layers.Input(shape=X[0].shape)
            x = self.embedding(inputs)
            x_1 = self.bigram(x)
            x_1 = layers.GlobalMaxPool1D()(x_1)
            x_2 = self.trigram(x)
            x_2 = layers.GlobalMaxPool1D()(x_2)
            x_3 = self.fourgram(x)
            x_3 = layers.GlobalMaxPool1D()(x_3)

            #merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
            merged = layers.concatenate([x_1, x_2, x_3], axis=-1)
            merged = self.dense_1(merged)
            merged = self.dropout(merged)
            output = self.last_dense(merged)
            model = keras.Model(inputs=[inputs], outputs=[output])
            return model
        
    VOCAB_SIZE = vocab_size #len(X[0][0])+1 # 65540

    EMB_DIM = 200
    NB_FILTERS = 100
    FFN_UNITS = 256
    NB_CLASSES = len(Y[0]) if type(Y[0]) == type([]) else 1

    DROPOUT_RATE = 0.2

    BATCH_SIZE = 32
    NB_EPOCHS = 5

    Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)
    Dcnn = Dcnn.get_model(X)
    Dcnn.summary()
    if NB_CLASSES == 1:
        Dcnn.compile(loss="mean_squared_error",
                    optimizer="adam",
                    metrics=["accuracy"])
    elif NB_CLASSES == 2:
        Dcnn.compile(loss="binary_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])
    else:
        Dcnn.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["categorical_accuracy"])
    #print(Dcnn.call(x_test[0:5], False))
    #Dcnn.summary()
    print("to train")
    Dcnn.fit(x_train,
         y_train,
         batch_size=BATCH_SIZE,
         epochs=NB_EPOCHS)
    print("to eval")
    results = Dcnn.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print(results)
    return Dcnn

def generate_result(text_to_evaluate, model, length_to_padding, dictionary):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", 
                                                 "morphologizer", 
                                                 #"parser", 
                                                 "senter", 
                                                 "attribute_ruler", 
                                                 "lemmatizer", 
                                                 "ner"
                                                 ])
    from gensim.corpora import Dictionary
    if type(dictionary) == Dictionary:
        doc = dictionary.doc2idx([token.text for token in nlp(text_to_evaluate)])
    else:
        doc = dictionary.encode(text_to_evaluate)
    print("doc: ", doc)
    doc = [x for x in doc if x != -1]
    print("doc: ", doc)
    X = padding([doc], length_to_padding)
    print("X padded: ", X)
    print("lenX0: ", len(X[0]))
    result = model.predict(X)
        
    return result

def save_data(model, length_to_padding, dictionary):
    import os
    if not os.path.exists("data/created_models/text_sentiment/"):
        os.mkdir("data/created_models/text_sentiment/")
    model.save("data/created_models/text_sentiment/neural_net.keras")
    from gensim.corpora import Dictionary
    if type(dictionary) == Dictionary:
        dictionary.save("data/created_models/text_sentiment/dictionary")
    else:
        pass #TODO: search a way to save
    with open("data/created_models/text_sentiment/param.txt", "w") as fobj:
        fobj.write(str(length_to_padding))
        fobj.flush()

def load_data(folder="data/created_models/text_sentiment/", using_gensim_dictionary=True):
    from tensorflow import keras
    from gensim.corpora import Dictionary
    model = keras.saving.load_model(os.path.join(folder, "neural_net.keras"))
    if using_gensim_dictionary:
        dictionary = Dictionary.load(os.path.join(folder, "dictionary"))
    else:
        pass #TODO: search a way to load
    with open(os.path.join(folder, "param.txt"), "r") as fobj:
        length_to_padding = int(fobj.read().strip())
    return model, length_to_padding, dictionary

if __name__ == "__main__":
    X , Y = get_data()
    X, Y = downsample(X, Y, 0.001)
    #For testing purposes
    #n_test = 100
    #import random
    #X = random.sample(X, n_test)
    #Y = random.sample(Y, n_test)
    #end testing code
    using_1_dim = True
    if not using_1_dim:
        Y, mappings = one_hot_encode(Y)
    else:
        mappings = None
        import numpy as np
        Y = np.asarray([float(y) for y in Y])
    print("Y OneHotEncoded, to clean")
    X = clean_data(X)
    print("cleaned, to tokenize")
    X, dictionary, nlp = tokenize_with_gensim_spacy(X)
    X = padding(X)
    #print("padded: ", X[0])
    #print("shapeX", X.shape)
    ##BoW
    #print("to bow")
    #X = idx_to_bow(X)
    #X, dictionary = tokenize_tfds(X)
    print("tokenized: ",X[0])
    #print("to pad")
    #X = padding(X)
    #import numpy as np
    #X = np.asarray(X)
    print("lenX0   : ", len(X[0]))
    length_to_padding = len(X[0])
    print("lenX0_10: ", X[0][10])
    print("shapeX  :", X.shape)  
    #print("Erróneos", sum([1 if len(x2)!= len(X[0]) else 0 for x in X for x2 in x]))
    model = create_run_copied_model(X, Y, len(X[0]))
    text_to_evaluate = "This draw is horrible, i'm so sad this thing is happening right now"
    result = generate_result(text_to_evaluate, model, length_to_padding, dictionary)
    print(f"result of evaluate: {text_to_evaluate}: ", result)
    print(mappings)
    #
    print("to save")
    save_data(model, length_to_padding, dictionary)
    print("to load")
    model, length_to_padding, dictionary = load_data()
    print("testing loaded data")
    result = generate_result(text_to_evaluate, model, length_to_padding, dictionary)
    print(f"result of evaluate: {text_to_evaluate}: ", result)
    print(mappings)
