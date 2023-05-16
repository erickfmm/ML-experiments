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
from load_data.loader.basic._glass import LoadGlass
from load_data.loader.basic._iris import LoadIris as LoadIrisBasic
from load_data.loader.basic._mnist_csv import LoadMnistCsv
from load_data.loader.basic._mnist_file import LoadMnist
from load_data.loader.basic.random_of_function import LoadRandom
from load_data.loader.basic._segment import LoadSegment
from load_data.loader.basic._titanic import LoadTitanic
from load_data.loader.basic._vehicle import LoadVehicle
from load_data.loader.basic.xortable import LoadXorTable



from load_data.loader.emotion_audio._aditya import LoadAditya
from load_data.loader.emotion_audio.berlin import LoadBerlin
from load_data.loader.emotion_audio.ravdess import LoadRavdess
from load_data.loader.emotion_audio.savee import LoadSavee
from load_data.loader.emotion_audio.tess import LoadTESS


from load_data.loader.emotion_eeg.individual_clasiff import LoadEEGIndividualEmotions
from load_data.loader.emotion_eeg._extracted_sqlite import LoadEEGEmotionExtracted


from load_data.loader.emotion_image._acmm10 import LoadACMM10ImageEmotion
from load_data.loader.emotion_image._hvei2016 import LoadHVEI2016ImageEmotion
from load_data.loader.emotion_image._oasis import LoadOASISImageEmotion





from load_data.loader.image._africa_fabric import LoadAfricaFabric
from load_data.loader.image._art_images_style import LoadArtImageStyle
from load_data.loader.image.butterfly_segment import LoadButterflySegment
from load_data.loader.image._Emotes import LoadTwitchEmotes





from load_data.loader.manga_anime.anime_faces import LoadAnimeFaces
from load_data.loader.manga_anime.anime_recomendations import LoadAnimeData
from load_data.loader.manga_anime._manga_adaptation import LoadMangaAdaptations


from load_data.loader.text.ENG_news_category import LoadNewsCategory
from load_data.loader.text._ENG_onu_debates import LoadOnuDebates
from load_data.loader.text._ENG_reddit_comments import LoadRedditComments
from load_data.loader.text._ENG_sarc2 import readbyparts_yielded
from load_data.loader.text.ENG_sarcasm_reddit_kaggle import LoadSarcasmRedditKaggle
from load_data.loader.text._ES_diccionarios import LoadDictionaryDuenasLerin
from load_data.loader.text._ES_diccionarios import LoadDictionaryHunspell
from load_data.loader.text._ES_diccionarios import LoadDictionaryRaeComplete
from load_data.loader.text._ES_names_spain import LoadNamesSpain
from load_data.loader.text._international_stopwords import LoadInternationalStopWords
from load_data.loader.text.stopwords import LoadStopwords

from load_data.loader.text._vectorized_words_txt_mongodb import VectorizedWordsTxtMongoDB



from load_data.loader.text_sentiment._ES_lex_ElhPolar_V1 import LoadElhPolarEs
from load_data.loader.text_sentiment.lexicons81langs import LoadLexicons81langs

from load_data.loader.translation.wmt import LoadWMT

from load_data.loader.videogames._pokemon_image_type import LoadPokemon

from load_data.loader._gsr_mahnob_emotion import LoadGsrMahnobEmotion
from load_data.loader._meteors import LoadMeteorsUnsupervised
from load_data.loader._recognition_human_actions_video import LoadRecognitionHumanActions
from load_data.loader._weedprices import LoadWeedPrices

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

print("stopwords 28")
l = LoadStopwords()
_ = l.get_all()

print("sentiment lexicons 81 langs")
l = LoadLexicons81langs()
_,_ = l.get_all()

print("wmt6")
l = LoadWMT()
_,_ = l.get_all()

print("end")