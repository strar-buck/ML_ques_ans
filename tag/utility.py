from sklearn.externals import joblib
import ast
import pandas as pd
import numpy as np
from collections import defaultdict
import collections

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.multioutput import MultiOutputClassifier                  # included from scikit-learn version 0.18.1 and onwards
from sklearn.ensemble import RandomForestClassifier
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics