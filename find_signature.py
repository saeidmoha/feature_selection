#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

#numpy.set_printoptions(threshold=numpy.inf)
#print(features_train)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train).toarray()
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]
#print(features_train)


### your code goes here

from sklearn import tree

clf = tree.DecisionTreeClassifier()

#t0 = time()
clf.fit(features_train, labels_train)
#print ("training time:", round(time()-t0, 3), "s")
    
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("accuracy is =", acc)
print("number of feautures: ", len(features_train[0])) # I added

feature_importances_ = clf.feature_importances_

print("feature_importances > 0.2 are: ")
""" worked but following is more elegent solution
j = 0
for i in feature_importances_:
    if (i > 0.2):
        print("number of feature is: ", j, "value is =", i)
    j += 1
"""
"""
import numpy as np
indices = np.argsort(feature_importances_)[::-1]
print ('Feature Ranking: ')
for i in range(10):
    print ("{} feature no.{} ({})".format(i+1,indices[i],feature_importances_[indices[i]]))

"""
for index, item in enumerate(feature_importances_):
    if item > 0.2:        
        print (index, item)       
       

feature_name = vectorizer.get_feature_names()
for index, item in enumerate(feature_name):
    if index == 33614 or index == 14343 or index == 21323:        
        print ("name[", index, "] = ", item) 


