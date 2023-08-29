import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..')))
######################################################

print("testing imports")
from load_data.loader.downloadable.boston_housing_keras import LoadBostonHousing
from load_data.loader.downloadable.cifar100_keras import LoadCifar100
from load_data.loader.downloadable.cifar10_keras import LoadCifar10
from load_data.loader.downloadable.fashion_mnist_keras import LoadFashionMnist
from load_data.loader.downloadable.iris_sklearn import LoadIris
from load_data.loader.downloadable.mnist_keras import LoadMnist


from load_data.loader.audio.spoken_digits import LoadSpokenDigits

from load_data.loader.basic.andtable import LoadAndTable
from load_data.loader.basic.empty import LoadEmpty
from load_data.loader.basic.random_of_function import LoadRandom
from load_data.loader.basic.xortable import LoadXorTable


from load_data.loader.emotion_audio.berlin import LoadBerlin
from load_data.loader.emotion_audio.ravdess import LoadRavdess
from load_data.loader.emotion_audio.savee import LoadSavee
from load_data.loader.emotion_audio.tess import LoadTESS


from load_data.loader.emotion_eeg.individual_clasiff import LoadEEGIndividualEmotions

from load_data.loader.image.oasis import LoadOASISImageEmotion

from load_data.loader.image.butterfly_segment import LoadButterflySegment
from load_data.loader.image.Emotes import LoadTwitchEmotes

from load_data.loader.manga_anime.anime_faces import LoadAnimeFaces
from load_data.loader.manga_anime.anime_recomendations import LoadAnimeData


from load_data.loader.text.ENG_news_category import LoadNewsCategory
from load_data.loader.text.ENG_onu_debates import LoadOnuDebates
from load_data.loader.text.ENG_sarcasm_reddit_kaggle import LoadSarcasmRedditKaggle
from load_data.loader.text.stopwords import LoadStopwords
from load_data.loader.text.ES_wikipedia_corpus import LoadES_Wikipedia_Corpus

from load_data.loader.text.vectorized_words_txt_mongodb import VectorizedWordsTxtMongoDB



from load_data.loader.text_sentiment.lexicons81langs import LoadLexicons81langs
from load_data.loader.text_sentiment.ENG_1million_tweets import Load1MTweets
from load_data.loader.text_sentiment.ENG_imdb_sentiment import LoadImdbSentiment
from load_data.loader.text_sentiment.ENG_RedditTwitterSentiment import LoadRedditOrTwitterSentiment
from load_data.loader.text_sentiment.ENG_Sentiment140 import LoadSentiment140
from load_data.loader.text_sentiment.ENG_Steam import LoadSteam


from load_data.loader.translation.wmt import LoadWMT

print("imported")

l = LoadAndTable()
_, _ = l.get_all()
l = LoadEmpty()
_ = l.get_all()
l = LoadRandom(50, 10, None)
_, _ = l.get_all()
l = LoadXorTable()
_, _ = l.get_all()


print("downloadable")


print("emotion audio")
print("berlin")
l = LoadBerlin()
_, _ = l.get_all()
print("ravdess")
l = LoadRavdess()
_, _ = l.get_all()
print("savee")
l = LoadSavee()
_, _ = l.get_all()
l = LoadTESS()
_, _ = l.get_all()

print("emotion eeg")
l = LoadEEGIndividualEmotions()
_ , _ = l.get_all()

print("butterfly")
l = LoadButterflySegment()
_ = l.get_all()

print("anime faces")
l = LoadAnimeFaces()
_ = l.get_all()

print("anime recomendations")
l = LoadAnimeData()
_, _ = l.get_all()

print("text")
print("news category")
l = LoadNewsCategory()
_,_ = l.get_all()

print("reddit sarcasm")
l = LoadSarcasmRedditKaggle()
_, _ = l.get_all()

#it works, but it lasts too much time loading ¿maybe load in parts?
#print("wikipedia spanish")
#l = LoadES_Wikipedia_Corpus()
#_ = l.get_all()

print("stopwords 28")
l = LoadStopwords()
_ = l.get_all()

print("sentiment lexicons 81 langs")
l = LoadLexicons81langs()
_,_ = l.get_all()

print("sentiment")

print("1M tweets")
l = Load1MTweets()
_,_ = l.get_all()
print("imdb")
l = LoadImdbSentiment()
_,_ = l.get_all()
print("reddit and twitter")
l = LoadRedditOrTwitterSentiment(source="reddit")
_,_ = l.get_all()
print("twitter")
l = LoadRedditOrTwitterSentiment(source="twitter")
_,_ = l.get_all()
print("sentiment 140")
l = LoadSentiment140()
_,_ = l.get_all()
print("steam")
l = LoadSteam()
_,_ = l.get_all()


print("wmt6")
l = LoadWMT()
_,_ = l.get_all()

print("onu debates")
l = LoadOnuDebates()
_ = l.get_all()

print("spoken digits")
l = LoadSpokenDigits()
_, _ = l.get_all()

print("oasis")
l = LoadOASISImageEmotion()
_, _ = l.get_all()

print("twitch emotes")
l = LoadTwitchEmotes()
_ = l.get_all()
print("end")