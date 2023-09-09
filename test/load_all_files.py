import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

print("testing imports")
from mlexperiments.load_data.loader.downloadable.boston_housing_keras import LoadBostonHousing
from mlexperiments.load_data.loader.downloadable.cifar100_keras import LoadCifar100
from mlexperiments.load_data.loader.downloadable.cifar10_keras import LoadCifar10
from mlexperiments.load_data.loader.downloadable.fashion_mnist_keras import LoadFashionMnist
from mlexperiments.load_data.loader.downloadable.iris_sklearn import LoadIris
from mlexperiments.load_data.loader.downloadable.mnist_keras import LoadMnist


from mlexperiments.load_data.loader.audio.spoken_digits import LoadSpokenDigits

from mlexperiments.load_data.loader.basic.andtable import LoadAndTable
from mlexperiments.load_data.loader.basic.empty import LoadEmpty
from mlexperiments.load_data.loader.basic.random_of_function import LoadRandom
from mlexperiments.load_data.loader.basic.xortable import LoadXorTable


from mlexperiments.load_data.loader.emotion_audio.berlin import LoadBerlin
from mlexperiments.load_data.loader.emotion_audio.ravdess import LoadRavdess
from mlexperiments.load_data.loader.emotion_audio.savee import LoadSavee
from mlexperiments.load_data.loader.emotion_audio.tess import LoadTESS


from mlexperiments.load_data.loader.emotion_eeg.individual_clasiff import LoadEEGIndividualEmotions

from mlexperiments.load_data.loader.image.oasis import LoadOASISImageEmotion

from mlexperiments.load_data.loader.image.butterfly_segment import LoadButterflySegment
from mlexperiments.load_data.loader.image.Emotes import LoadTwitchEmotes

from mlexperiments.load_data.loader.manga_anime.anime_faces import LoadAnimeFaces
from mlexperiments.load_data.loader.manga_anime.anime_recomendations import LoadAnimeData


from mlexperiments.load_data.loader.text.ENG_news_category import LoadNewsCategory
from mlexperiments.load_data.loader.text.ENG_onu_debates import LoadOnuDebates
from mlexperiments.load_data.loader.text.ENG_sarcasm_reddit_kaggle import LoadSarcasmRedditKaggle
from mlexperiments.load_data.loader.text.stopwords import LoadStopwords
from mlexperiments.load_data.loader.text.ES_wikipedia_corpus import LoadES_Wikipedia_Corpus

from mlexperiments.load_data.loader.text.vectorized_words_txt_mongodb import VectorizedWordsTxtMongoDB



from mlexperiments.load_data.loader.text_sentiment.lexicons81langs import LoadLexicons81langs
from mlexperiments.load_data.loader.text_sentiment.ENG_1million_tweets import Load1MTweets
from mlexperiments.load_data.loader.text_sentiment.ENG_imdb_sentiment import LoadImdbSentiment
from mlexperiments.load_data.loader.text_sentiment.ENG_RedditTwitterSentiment import LoadRedditOrTwitterSentiment
from mlexperiments.load_data.loader.text_sentiment.ENG_Sentiment140 import LoadSentiment140
from mlexperiments.load_data.loader.text_sentiment.ENG_Steam import LoadSteam


from mlexperiments.load_data.loader.translation.wmt import LoadWMT

print("imported")

l = LoadAndTable()
_, _ = l.get_X_Y()
l = LoadEmpty()
_ = l.get_X_Y()
l = LoadRandom(50, 10, None)
_, _ = l.get_X_Y()
l = LoadXorTable()
_, _ = l.get_X_Y()


print("downloadable")


print("emotion audio")
print("berlin")
l = LoadBerlin()
_, _ = l.get_X_Y()
print("ravdess")
l = LoadRavdess()
_, _ = l.get_X_Y()
print("savee")
l = LoadSavee()
_, _ = l.get_X_Y()
l = LoadTESS()
_, _ = l.get_X_Y()

print("emotion eeg")
l = LoadEEGIndividualEmotions()
_ , _ = l.get_X_Y()

print("butterfly")
l = LoadButterflySegment()
_ = l.get_all()

print("anime faces")
l = LoadAnimeFaces()
_ = l.get_data()

print("anime recomendations")
l = LoadAnimeData()
_, _ = l.get_X_Y()

print("text")
print("news category")
l = LoadNewsCategory()
_,_ = l.get_X_Y()

print("reddit sarcasm")
l = LoadSarcasmRedditKaggle()
_, _ = l.get_X_Y()

#it works, but it lasts too much time loading ¿maybe load in parts?
#print("wikipedia spanish")
#l = LoadES_Wikipedia_Corpus()
#_ = l.get_all()

print("stopwords 28")
l = LoadStopwords()
_ = l.get_data()

print("sentiment lexicons 81 langs")
l = LoadLexicons81langs()
_,_ = l.get_X_Y()

print("sentiment")

print("1M tweets")
l = Load1MTweets()
_,_ = l.get_X_Y()
print("imdb")
l = LoadImdbSentiment()
_,_ = l.get_X_Y()
print("reddit and twitter")
l = LoadRedditOrTwitterSentiment(source="reddit")
_,_ = l.get_X_Y()
print("twitter")
l = LoadRedditOrTwitterSentiment(source="twitter")
_,_ = l.get_X_Y()
print("sentiment 140")
l = LoadSentiment140()
_,_ = l.get_X_Y()
print("steam")
l = LoadSteam()
_,_ = l.get_X_Y()


print("wmt6")
l = LoadWMT()
_,_ = l.get_data()

print("onu debates")
l = LoadOnuDebates()
_ = l.get_data()

print("spoken digits")
l = LoadSpokenDigits()
_, _ = l.get_X_Y()

print("oasis")
l = LoadOASISImageEmotion()
_, _ = l.get_X_Y()


print("twitch emotes")
l = LoadTwitchEmotes()
_ = l.get_data()


print("end")