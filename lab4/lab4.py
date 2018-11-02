#!/usr/bin/env python
# coding: utf-8

# In[527]:


import pandas as pd
import nltk
import glob
import sys
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[528]:


#reading all the files for training using pandas and concatenating data from all the training files
all_files = ["imdb_labelled.txt","amazon_cells_labelled.txt","yelp_labelled.txt"]
dataframe = pd.concat(pd.read_csv(file, sep='\t', names = ['txt', 'label'],index_col=None, header=0) for file in all_files)


# In[529]:


def trainNormNB():
    
    # Using TFIDF, short for term frequency–inverse document frequency
    # This transforms text to feature vectors
    
    # Creating a normalized version of Naive Bayes Classifier
    # Normalisation of text is done here
    stop_set = set(stopwords.words('english'))
    norm_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stop_set)

    y = dataframe.label    #setting dependent variables, the labels 0 for negative and 1 for positive sentiment
    x = norm_vectorizer.fit_transform(dataframe.txt)    #transforming data in the dataframe to features from text

    # Training testing split
    # Using a random state to guarantee the same results whenever training is done
    x_training, x_testing, y_training, y_testing = train_test_split(x, y, random_state=29, train_size = 0.95, test_size = 0.05)

    # Training the normalized Naive Bayes Classifier
    norm_nbClassifier = naive_bayes.MultinomialNB()
    norm_nbClassifier.fit(x_training, y_training)
    
    return norm_nbClassifier, norm_vectorizer


# In[530]:


def trainUnNormNB():
    # Using TFIDF, short for term frequency–inverse document frequency
    # This transforms text to feature vectors

    # Creating an unnormalized version of Naive Bayes Classifier
    unNorm_vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, strip_accents=None)

    unNorm_y = dataframe.label    #setting dependent variables, the labels 0 for negative and 1 for positive sentiment
    unNorm_x = unNorm_vectorizer.fit_transform(dataframe.txt)    #transforming data in the dataframe to features from text

    # Training testing split
    # Using a random state to guarantee the same results whenever training is done
    unNorm_x_training, unNorm_x_testing, unNorm_y_training, unNorm_y_testing = train_test_split(unNorm_x, unNorm_y, random_state=29, train_size = 0.95, test_size = 0.05)

    # Training the unnormalized Naive Bayes Classifier
    unNorm_nbClassifier = naive_bayes.MultinomialNB()
    unNorm_nbClassifier.fit(unNorm_x_training, unNorm_y_training)
    
    return unNorm_nbClassifier, unNorm_vectorizer


# In[531]:


def testNormNB(testfile):
    df = pd.read_csv(testfile,sep='\t', names = ['txt', 'label'],index_col=None, header=-1)
    
    norm_nbClassifier, norm_vectorizer = trainNormNB()
    
    y = list(df.label)    #setting dependent variables, the labels 0 for negative and 1 for positive sentiment
    x = df.txt    #setting the independent variables of features from text
    
    predict_list = []
    
    for i in range(len(x)):
        sentiment = np.array([str(x[i])])
        sentiment_tranform = norm_vectorizer.transform(sentiment)
        prediction = norm_nbClassifier.predict(sentiment_tranform)
        
        predict_list.append(prediction[0])
    
    precision, recall, accuracy, f1measure = contingencyTable(predict_list,y)
    
    return predict_list, precision, recall, accuracy, f1measure


# In[532]:


def testUnNormNB(testfile):
    df = pd.read_csv(testfile,sep='\t', names = ['txt', 'label'],index_col=None, header=-1)
    
    unNorm_nbClassifier, unNorm_vectorizer = trainUnNormNB()
    
    y = list(df.label)    #setting dependent variables, the labels 0 for negative and 1 for positive sentiment
    x = df.txt    #setting the independent variables of features from text
    
    predict_list = []
    
    for i in range(len(x)):
        sentiment = np.array([str(x[i])])
        sentiment_tranform = unNorm_vectorizer.transform(sentiment)
        prediction = unNorm_nbClassifier.predict(sentiment_tranform)
        
        predict_list.append(prediction[0])
    
    precision, recall, accuracy, f1measure = contingencyTable(predict_list,y)
    
    return predict_list, precision, recall, accuracy, f1measure


# In[533]:


def trainNormLR():
    
    # Using TFIDF, short for term frequency–inverse document frequency
    # This transforms text to feature vectors
    
    # Creating a normalized version of Logistical Regression Classifier
    # Normalisation of text is done here
    stop_set = set(stopwords.words('english'))
    norm_vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stop_set)

    y = dataframe.label    #setting dependent variables, the labels 0 for negative and 1 for positive sentiment
    x = norm_vectorizer.fit_transform(dataframe.txt)    #transforming data in the dataframe to features from text

    # Training testing split
    # Using a random state to guarantee the same results whenever training is done
    x_training, x_testing, y_training, y_testing = train_test_split(x, y, random_state=29, train_size = 0.95, test_size = 0.05)

    # Training the normalized Naive Bayes Classifier
    norm_lrClassifier = LogisticRegression()
    norm_lrClassifier.fit(x_training, y_training)
    
    return norm_lrClassifier, norm_vectorizer


# In[534]:


def trainUnNormLR():
    # Using TFIDF, short for term frequency–inverse document frequency
    # This transforms text to feature vectors

    # Creating an unnormalized version of Naive Bayes Classifier
    unNorm_vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, strip_accents=None)

    unNorm_y = dataframe.label    #setting dependent variables, the labels 0 for negative and 1 for positive sentiment
    unNorm_x = unNorm_vectorizer.fit_transform(dataframe.txt)    #transforming data in the dataframe to features from text

    # Training testing split
    # Using a random state to guarantee the same results whenever training is done
    unNorm_x_training, unNorm_x_testing, unNorm_y_training, unNorm_y_testing = train_test_split(unNorm_x, unNorm_y, random_state=29, train_size = 0.95, test_size = 0.05)

    # Training the unnormalized Naive Bayes Classifier
    unNorm_lrClassifier = LogisticRegression()
    unNorm_lrClassifier.fit(unNorm_x_training, unNorm_y_training)
    
    return unNorm_lrClassifier, unNorm_vectorizer


# In[535]:


def testNormLR(testfile):
    df = pd.read_csv(testfile,sep='\t', names = ['txt', 'label'],index_col=None, header=-1)
    
    norm_lrClassifier, norm_vectorizer = trainNormLR()
    
    y = list(df.label)    #setting dependent variables, the labels 0 for negative and 1 for positive sentiment
    x = df.txt    #setting the independent variables of features from text
    
    predict_list = []
    
    for i in range(len(x)):
        sentiment = np.array([str(x[i])])
        sentiment_tranform = norm_vectorizer.transform(sentiment)
        prediction = norm_lrClassifier.predict(sentiment_tranform)
        
        predict_list.append(prediction[0])
    
    precision, recall, accuracy, f1measure = contingencyTable(predict_list,y)
    
    return predict_list, precision, recall, accuracy, f1measure


# In[536]:


def testUnNormLR(testfile):
    df = pd.read_csv(testfile,sep='\t', names = ['txt', 'label'],index_col=None, header=-1)
    
    unNorm_lrClassifier, unNorm_vectorizer = trainUnNormLR()
    
    y = list(df.label)    #setting dependent variables, the labels 0 for negative and 1 for positive sentiment
    x = df.txt    #setting the independent variables of features from text
    
    predict_list = []
    
    for i in range(len(x)):
        sentiment = np.array([str(x[i])])
        sentiment_tranform = unNorm_vectorizer.transform(sentiment)
        prediction = unNorm_lrClassifier.predict(sentiment_tranform)
        
        predict_list.append(prediction[0])
    
    precision, recall, accuracy, f1measure = contingencyTable(predict_list,y)
    
    return predict_list, precision, recall, accuracy, f1measure


# In[537]:


def contingencyTable(systemLabels, goldLabels):    # Building a 2 dimensional contingency table
    
    table = [[0, 0],[0, 0]]    # The first inner list represents the true and false positives [tp and fp]
                                            # The seccond inner lists represents the false and true negatives [fn and tn]
        
    for i in range(len(systemLabels)):
        if systemLabels[i] == 1 and goldLabels[i] == 1: #adding to true possitive
            table[0][0] += 1
        elif systemLabels[i] == 0 and goldLabels[i] == 0: #adding to true negative
            table[1][1] += 1
        elif systemLabels[i] == 1 and goldLabels[i] == 0: #adding to false positive
            table[0][1] += 1
        else:
            table[1][0] += 1  #adding to false negative
    
    tp = table[0][0]
    fp = table[0][1]
    fn = table[1][0]
    tn = table[1][1]
    
    precision = round(tp/(tp+fp),2)
    recall = round(tp/(tp+fn),2)
    accuracy = round((tp+tn)/(tp+fp+tn+fn),2)
    f1measure = round((2*precision*recall)/(precision+recall),2)  #using a weight of 1 for the f-measure
    
    return precision, recall, accuracy, f1measure


# In[538]:


def createResultsFile(classifierType, version, systemLabels, precision, recall, accuracy, f1measure):
    
    out_file = open('results-'+classifierType+'-'+version+'.txt', 'w')
    out_file.write("System Labels:" + "\n")
    
    for label in systemLabels:
        out_file.write(str(label) + "\n")
    
    out_file.write("\t"+ '[0.0 - 1.0]' + "\n")  
    out_file.write("Accuracy: "+ str(accuracy) +"\n")
    out_file.write("Precision: "+ str(precision) +"\n")
    out_file.write("Recall: "+ str(recall) +"\n")
    out_file.write("F1-measure: "+ str(f1measure) +"\n")


# In[540]:


if __name__ == '__main__':
    
    classifier_type = sys.argv[1]
    version = sys.argv[2]
    testfile = sys.argv[3]
    
    if classifier_type == 'nb' and version == 'n':
        predict_list, precision, recall, accuracy, f1measure = testNormNB(testfile)
    elif classifier_type == 'nb' and version == 'u':
        predict_list, precision, recall, accuracy, f1measure = testUnNormNB(testfile)
    elif classifier_type == 'lr' and version == 'n':
        predict_list, precision, recall, accuracy, f1measure = testNormLR(testfile)
    elif classifier_type == 'lr' and version == 'u':
        predict_list, precision, recall, accuracy, f1measure = testUnNormLR(testfile)
    else:
        print("Wrong intput combination. < E.g. “python lab4.py nb u test.txt”" + "\n"
              +"should run the unnormalized version of your naïve bayes classifier" + "\n"
              +" on a file called test.txt.>")
        sys.exit()
        
    createResultsFile(classifier_type, version, predict_list, precision, recall, accuracy, f1measure)

