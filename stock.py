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

data = pd.read_csv("Combined_News_DJIA.csv")
data.head()

data.set_index('Date')

X = data.iloc[0:,2:].astype(str)
Y = data['Label']

## Text Process
stopword = stopwords.words('english')
snowball = SnowballStemmer('english')
# remove punctuation, stopword and convert str into lowercase
for k in range(len(X.columns)):
    for i in range(len(X)):
        X.iloc[i,k] = ' '.join(word.lower() for word in nltk.word_tokenize(X.iloc[i,k]) 
                            if word not in punctuation and word not in stopword)

# stem
for k in range(len(X.columns)):
    for i in range(len(X)):
        X.iloc[i,k] = ' '.join(snowball.stem(word) for word in nltk.word_tokenize(X.iloc[i,k]))

# remove non-essential character b, ``, ' which are overlooked by last step
for k in range(len(X.columns)):
    for i in range(len(X)):
        X.iloc[i,k] = X.iloc[i,k].replace('``', '').replace('b','').replace("'", '')

X.columns = range(1,26)
X['date'] = data['Date']
review_data = X.set_index('date')

review_data.to_csv(r'textProcess.csv')

## Modeling
X_text = pd.read_csv('textProcess.csv', usecols=range(1,26))
X_text

# # create vectorizer for each feature 1:25
# count_vectorize = CountVectorizer()
# # use loop to let vectorizer to learn token for 25 features
# # set a dict to store all the vectorizers
# text = {}
# for i in range(1,len(X_text.columns)+1):
#     text['x{0}'.format(i)] = count_vectorize.fit_transform(X_text[str(i)].apply(lambda x: np.str_(x)))

# ## SVM - inital
# # create model
# svm = SVC()
# # get 5-fold cv and use loop to let model learn each feature
# accs = {}
# score = {}
# for i in range(1,len(text)+1): 
#     score['{0}'.format(i)] = cross_val_score(svm, text['x'+str(i)], Y, scoring='accuracy', cv=5)
#     accs['x{0}'.format(i)] = np.round(np.mean(score['{0}'.format(i)]),6)

# # create vectorizer
# count_vectorizer = CountVectorizer()

# x = {}
# for i in range(1, len(X_text.columns)+1):
#     x['{0}'.format(i)] = pd.DataFrame(count_vectorizer.fit_transform(X_text[str(i)].apply(lambda x: np.str_(x))).todense(), 
#     columns=count_vectorizer.get_feature_names())


# newData = pd.concat(
#     [x[df] for df in x], axis=1
# )

# svm = SVC()
# accs = cross_val_score(svm, newData, Y, scoring='accuracy', cv=5)
# print('Accuracy of svm model: %s' %(np.round(np.mean(accs),3)))

X_train = {}
X_test = {}
for i in range(1, len(X_text.columns)+1):
    x_train, x_test, y_train, y_test = train_test_split(X_text[str(i)], Y, train_size=0.7)
    x_train = x_train.apply(lambda x: np.str_(x))
    x_test = x_test.apply(lambda x: np.str_(x))
    cv = CountVectorizer(stop_words='english').fit(x_train)
    X_train['{0}'.format(i)] = pd.DataFrame(cv.transform(x_train).todense(), columns=cv.get_feature_names())
    X_test['{0}'.format(i)] = pd.DataFrame(cv.transform(x_test).todense(), columns=cv.get_feature_names())

train = pd.concat(
    [X_train[key] for key in X_train], axis=1
)

test = pd.concat(
    [X_test[key] for key in X_test], axis=1
)

lr = LogisticRegression()
lr.fit(train, y_train)
lr.score(test, y_test)

svm = SVC()
svm.fit(train, y_train)
svm.score(test, y_test)

accs = cross_val_score(svm, train, y_train, cv=10, scoring='accuracy')
print('Accuracy of svm model: %s' %(np.round(np.mean(accs),3)))

## modeling
# random forest
max_depths = range(10,110,10) # the maximum depth of each tree
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
alphas = [0,0.1,0.25,0.5,0.75,1]
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
    'alpha': [0]
}
mnb = MultinomialNB()
grid_mnb1 = GridSearchCV(mnb, param_grid=param_mnb,cv=10)
grid_mnb1.fit(train, y_train)
grid_mnb1.best_score_



