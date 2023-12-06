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


def downsample(X, Y):
    print("to downsample")
    n = int(min([sum([1 if x==0 else 0 for x in Y]), 
            sum([1 if x==0.5 else 0 for x in Y]), 
            sum([1 if x==1 else 0 for x in Y])]) * 0.8)

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
    return Y


def clean_data(X):
    from mlexperiments.preprocessing.text.clean import clean_tweet
    X = [clean_tweet(x) for x in X]
    return X

def tokenize(X):
    #tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #    data_clean, target_vocab_size=2**16
    #)
    #data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]
    return X #TODO: implement

def padding(X):
    #MAX_LEN = max([len(sentence) for sentence in data_inputs])
    #data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs,
    #                                                        value=0,
    #                                                        padding="post",
    #                                                        maxlen=MAX_LEN)
    return X #TODO: implement



if __name__ == "__main__":
    X , Y = get_data()
    X, Y = downsample(X, Y)
    Y = one_hot_encode(Y)
    X = clean_data(X)
    X = tokenize(X)
    X = padding(X)
