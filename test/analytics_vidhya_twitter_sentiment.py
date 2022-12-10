

# Let’s load the libraries which will used in this course.
print("to import")
import re    # for regular expressions 
import nltk  # for text manipulation 
import string 
#import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  

print("all imported")
pd.set_option("display.max_colwidth", 200) 
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# %matplotlib inline

# Let’s read train and test datasets. Download data from here.

from os.path import join
folder = "train_data/Folder_NLPEnglsh_Sentiment/analyticsvidhya twitter"

print("loading train")
train = pd.read_csv(join(folder, 'train_E6oV3lV.csv'))
print("loading test")
test = pd.read_csv(join(folder, 'test_tweets_anuFYb8.csv'))

print("data loaded")
