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
from mpl_toolkits import mplot3d

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
import sklearn.metrics as sm
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Import vectorizers to turn text into numeric
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 


# import data
dataX = pd.read_csv('textProcess.csv')
dataX.head()
X = dataX.set_index('date')
dataY = pd.read_csv("Combined_News_DJIA.csv")
dataY.head()
Y = dataY['Label']

# join all the columns together
X = X.applymap(str)
X = X.iloc[:, :].apply(lambda x: ' '.join(x), axis=1)

## modeling
# convert to unicode string
X_text = X.apply(lambda x: np.str_(x))

# create vectorizer
count_vectorizer = CountVectorizer()
# let vectorizer learn tokens
count_vectorizer.fit(X_text)
# convert string to numeric
text = count_vectorizer.transform(X_text)

# train test split
X[:'12/31/2014'].count
X_train = text[:1611]
X_test = text[1611:] 
y_train = Y[:1611]
y_test = Y[1611:]

# Naive Bayes
alphas = range(1,100)
accuracies_mnb = []
for alpha in alphas:
    param_mnb = {
        'alpha': [alpha]
    }
    mnb = MultinomialNB()
    grid_mnb = GridSearchCV(mnb, param_grid=param_mnb, scoring='accuracy')
    grid_mnb.fit(X_train, y_train)
    accuracies_mnb.append(grid_mnb.best_score_)

plt.plot(alphas, accuracies_mnb)
plt.show()

# random forest
max_depths = [int(x) for x in np.linspace(10,100,10)] # the maximum depth of each tree
n_estimators = [int(x) for x in np.linspace(10,100,10)] # Number of trees in random forest
param_rf = {
    'max_depth': max_depths,
    'n_estimators': n_estimators
}
## create empty unlearned random forest model
random_forest = RandomForestClassifier()
## tune the model and get 10-fold cross validation results 
grid_rf = GridSearchCV(random_forest, param_grid=param_rf, scoring='AUC')
## fit and train the random forest
grid_rf.fit(X_train,y_train)
## get cv results 
grid_rf.best_score_

# tuning hyper random forest
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Number of trees in random forest
n_estimators = [grid_rf.best_params_['n_estimators']]
# Maximum number of levels in tree
max_depth = [grid_rf.best_params_['max_depth']]
# hyperparameters tuning
param_rf_tune = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'max_features': max_features,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}
# create empty unlearned random forest model
random_forest = RandomForestClassifier()
# tune the model and get 10-fold cross validation results 
grid_rf1 = GridSearchCV(random_forest, param_grid=param_rf_tune)
# fit and train the random forest
grid_rf1.fit(X_train,y_train)
# get the best accuracy
grid_rf1.best_score_


# evaluation
# model - rf
pred = grid_mnb.predict(X_test)
probs = grid_mnb.predict_proba(X_test)[:,1]
roc_value = roc_auc_score(y_test, probs)



