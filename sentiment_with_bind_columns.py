import csv
import os

# import pandas and numpy
import pandas as pd
import numpy as np

# import nltk adn string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
import string
from string import punctuation
from nltk.stem import SnowballStemmer

# import plotting
import matplotlib.pylab as plt
import seaborn as sns

# Import models and evaluation functions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Import vectorizers to turn text into numeric
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 


# import data
dataX = pd.read_csv('textProcess.csv', header=None, usecols=range(1,26), skiprows=1)
dataX.head()
dataY = pd.read_csv("Combined_News_DJIA.csv")
dataY.head()
Y = dataY['Label']
Y

# append all the columns to the bottom of column 1
col1 = dataX[1]
for i in range(2,len(dataX.columns)+1):
    col1 = col1.append(dataX[i])
x = pd.DataFrame(col1)
x.shape
x.columns = ['review']

# repeat Y 25 times to comply with X
y = np.tile(Y, (1,25)).transpose()
y = pd.DataFrame(y)
y.shape
y.columns = ['target']

## modeling
# train test split
X_train, X_test, y_train, y_test = train_test_split(x['review'], y['target'], train_size=0.7)

# convert to unicode string
X_train = X_train.apply(lambda x: np.str_(x))
X_test = X_test.apply(lambda x: np.str_(x))

# create vectorizer 
count_vectorizer = CountVectorizer(ngram_range=(1,3))
# let model learn token
count_vectorizer.fit(X_train)
# convert string to numeric
train = count_vectorizer.transform(X_train)  

# random forest
max_depths = range(100,1100, 100) # the maximum depth of each tree
accuracies_rf = []
for max_depth in max_depths: 
    param_rf = {
        'max_depth': [max_depth],
        'n_estimators': [10]
    }
    ## create empty unlearned random forest model
    random_forest = RandomForestClassifier()
    ## tune the model and get 10-fold cross validation results 
    grid_rf = GridSearchCV(random_forest, param_grid=param_rf)
    ## fit and train the random forest
    grid_rf.fit(train,y_train)
    ## get cv results and keep tracking them
    accuracies_rf.append(grid_rf.best_score_)

## plot the results
plt.plot(max_depths, accuracies_rf)
plt.show()

# Naive Bayes
alphas = range(1,100)
accuracies_mnb = []
for alpha in alphas:
    param_mnb = {
        'alpha': [alpha]
    }
    mnb = MultinomialNB()
    grid_mnb = GridSearchCV(mnb, param_grid=param_mnb)
    grid_mnb.fit(train, y_train)
    accuracies_mnb.append(grid_mnb.best_score_)

plt.plot(alphas, accuracies_mnb)
plt.show()

# Naive Bayes with cv = 10 & alphas = 0 
param_mnb = {
    'alpha': [grid_mnb.best_params_['alpha']]
}
mnb = MultinomialNB()
grid_mnb1 = GridSearchCV(mnb, param_grid=param_mnb,cv=50)
grid_mnb1.fit(train, y_train)
grid_mnb1.best_score_










