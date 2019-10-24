

#Let’s load the libraries which will used in this course.

import re    # for regular expressions 
import nltk  # for text manipulation 
import string 
#import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  

pd.set_option("display.max_colwidth", 200) 
#warnings.filterwarnings("ignore", category=DeprecationWarning) 

#%matplotlib inline

#Let’s read train and test datasets. Download data from here.

from os.path import join
folder = "train_data\\not_shared\\Folder_Twitter\\analyticsvidhya twitter"

train  = pd.read_csv(join(folder, 'train_E6oV3lV.csv')) 
test = pd.read_csv(join(folder, 'test_tweets_anuFYb8.csv'))

