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
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    # Build vocabulary from corpus
    vocab = {"[PAD]": 0, "[UNK]": 1}
    idx = 2
    for sentence in X:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    tokenizer.model = WordLevel(vocab, unk_token="[UNK]")
    X_encoded = [tokenizer.encode(sentence).ids for sentence in X]
    return X_encoded, tokenizer

def padding(X, max_len :int = None):
    import numpy as np
    if max_len is None:
        max_len = max([len(sentence) for sentence in X])
    padded = np.zeros((len(X), max_len), dtype=np.int32)
    for i, seq in enumerate(X):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded

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
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class DCNN(nn.Module):

        def __init__(self,
                    vocab_size,
                    emb_dim=128,
                    nb_filters=50,
                    FFN_units=512,
                    nb_classes=2,
                    dropout_rate=0.1,
                    name="dcnn"):
            super(DCNN, self).__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim)
            self.bigram = nn.Conv1d(in_channels=emb_dim, out_channels=nb_filters, kernel_size=2, padding="valid")
            self.trigram = nn.Conv1d(in_channels=emb_dim, out_channels=nb_filters, kernel_size=3, padding="valid")
            self.fourgram = nn.Conv1d(in_channels=emb_dim, out_channels=nb_filters, kernel_size=4, padding="valid")
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.dense_1 = nn.Linear(3 * nb_filters, FFN_units)
            self.dropout = nn.Dropout(p=dropout_rate)
            if nb_classes <= 2:
                self.last_dense = nn.Linear(FFN_units, 1)
            else:
                self.last_dense = nn.Linear(FFN_units, nb_classes)
            self.nb_classes = nb_classes

        def forward(self, x):
            # x: (batch_size, seq_len)
            x = self.embedding(x)          # (batch_size, seq_len, emb_dim)
            x = x.permute(0, 2, 1)         # (batch_size, emb_dim, seq_len) for Conv1d

            x_1 = torch.relu(self.bigram(x))   # (batch_size, nb_filters, *)
            x_1 = self.pool(x_1).squeeze(-1)   # (batch_size, nb_filters)
            x_2 = torch.relu(self.trigram(x))
            x_2 = self.pool(x_2).squeeze(-1)
            x_3 = torch.relu(self.fourgram(x))
            x_3 = self.pool(x_3).squeeze(-1)

            merged = torch.cat([x_1, x_2, x_3], dim=-1)  # (batch_size, 3 * nb_filters)
            merged = torch.relu(self.dense_1(merged))
            merged = self.dropout(merged)
            output = self.last_dense(merged)
            if self.nb_classes <= 2:
                output = torch.sigmoid(output)
            else:
                output = torch.softmax(output, dim=-1)
            return output

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1992)

    VOCAB_SIZE = vocab_size
    EMB_DIM = 200
    NB_FILTERS = 100
    FFN_UNITS = 256
    NB_CLASSES = len(y_train[0]) if isinstance(y_train[0], (list, np.ndarray)) and len(y_train[0]) > 1 else 1

    DROPOUT_RATE = 0.2
    BATCH_SIZE = 256
    NB_EPOCHS = 25

    Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)
    print(Dcnn)

    # Prepare data
    x_train_t = torch.tensor(np.array(x_train), dtype=torch.long)
    x_test_t = torch.tensor(np.array(x_test), dtype=torch.long)
    if NB_CLASSES == 1:
        y_train_t = torch.tensor(np.array(y_train, dtype=np.float32)).unsqueeze(1)
        y_test_t = torch.tensor(np.array(y_test, dtype=np.float32)).unsqueeze(1)
    else:
        y_train_t = torch.tensor(np.array(y_train, dtype=np.float32))
        y_test_t = torch.tensor(np.array(y_test, dtype=np.float32))

    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(x_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    if NB_CLASSES <= 2:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Dcnn.parameters())

    # Train
    print("to train")
    Dcnn.train()
    for epoch in range(NB_EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = Dcnn(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            if NB_CLASSES <= 2:
                preds = (outputs >= 0.5).float()
                correct += (preds == batch_y).sum().item()
            else:
                _, preds = torch.max(outputs, 1)
                _, targets = torch.max(batch_y, 1)
                correct += (preds == targets).sum().item()
            total += len(batch_y)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{NB_EPOCHS} - loss: {total_loss/len(x_train_t):.4f} - acc: {correct/total:.4f}")

    # Evaluate
    print("to eval")
    Dcnn.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = Dcnn(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * len(batch_x)
            if NB_CLASSES <= 2:
                preds = (outputs >= 0.5).float()
                correct += (preds == batch_y).sum().item()
            else:
                _, preds = torch.max(outputs, 1)
                _, targets = torch.max(batch_y, 1)
                correct += (preds == targets).sum().item()
            total += len(batch_y)
    print(f"Test loss: {total_loss/len(x_test_t):.4f} - Test acc: {correct/total:.4f}")
    return Dcnn

def generate_result(text_to_evaluate, model, length_to_padding, dictionary):
    import numpy as np
    import torch
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
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.long)
        result = model(X_t).numpy()
        
    return result

def save_data(model, length_to_padding, dictionary):
    import os
    import torch
    if not os.path.exists("data/created_models/text_sentiment/"):
        os.mkdir("data/created_models/text_sentiment/")
    torch.save(model.state_dict(), "data/created_models/text_sentiment/neural_net.pt")
    from gensim.corpora import Dictionary
    if type(dictionary) == Dictionary:
        dictionary.save("data/created_models/text_sentiment/dictionary")
    else:
        pass #TODO: search a way to save
    with open("data/created_models/text_sentiment/param.txt", "w") as fobj:
        fobj.write(str(length_to_padding))
        fobj.flush()

def load_data(folder="data/created_models/text_sentiment/", using_gensim_dictionary=True):
    import torch
    from gensim.corpora import Dictionary
    # Need to know model params to reconstruct; load with weights_only=False for flexibility
    state_dict = torch.load(os.path.join(folder, "neural_net.pt"), weights_only=True)
    # Infer vocab_size from embedding layer
    vocab_size = state_dict["embedding.weight"].shape[0]
    emb_dim = state_dict["embedding.weight"].shape[1]
    nb_filters = state_dict["bigram.weight"].shape[0]
    ffn_units = state_dict["dense_1.weight"].shape[0]
    nb_classes_raw = state_dict["last_dense.weight"].shape[0]
    if nb_classes_raw == 1:
        nb_classes = 2  # binary
    else:
        nb_classes = nb_classes_raw
    model = DCNN(vocab_size=vocab_size,
                 emb_dim=emb_dim,
                 nb_filters=nb_filters,
                 FFN_units=ffn_units,
                 nb_classes=nb_classes)
    model.load_state_dict(state_dict)
    model.eval()
    if using_gensim_dictionary:
        dictionary = Dictionary.load(os.path.join(folder, "dictionary"))
    else:
        pass #TODO: search a way to load
    with open(os.path.join(folder, "param.txt"), "r") as fobj:
        length_to_padding = int(fobj.read().strip())
    return model, length_to_padding, dictionary

def training_pipeline():
    X , Y = get_data()
    X, Y = downsample(X, Y, 0.8)
    #For testing purposes
    #n_test = 100
    #import random
    #X = random.sample(X, n_test)
    #Y = random.sample(Y, n_test)
    #end testing code
    using_1_dim = False
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
    model = create_run_copied_model(X, Y, max([x for doc in X for x in doc])+1)
    text_to_evaluate = "I hate to be with these people in the funeral"
    result = generate_result(text_to_evaluate, model, length_to_padding, dictionary)
    print(f"result of evaluate: {text_to_evaluate}: ", result)
    print(mappings)
    #
    print("to save")
    save_data(model, length_to_padding, dictionary)

def evaluation_pipeline():
    print("to load")
    model, length_to_padding, dictionary = load_data()
    print("testing loaded data")
    while True:
        text_to_evaluate = input("Ingrese un texto para evaluar: ")
        result = generate_result(text_to_evaluate, model, length_to_padding, dictionary)
        print(f"result of evaluate: {text_to_evaluate}: ", result)
        if text_to_evaluate == "exit":
            break

if __name__ == "__main__":
    choose_pipeline = int(input("Ingrese el pipeline a ejecutar (1 para entrenar, y 2 para evaluar): "))
    if choose_pipeline == 1:
        training_pipeline()
    elif choose_pipeline == 2:
        evaluation_pipeline()
    else:
        import sys
        sys.exit(0)    
    
