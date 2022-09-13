import Eval_Matrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import KFold


#import time



# a dummy function that just returns its input
def identity(x):
    return x

# decide on TF-IDF vectorization for feature
# based on the value of tfidf (True/False)
def tf_idf_func(tfidf):
    # TODO - change the values
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)

    else:
        vec = CountVectorizer(preprocessor = identity, tokenizer = identity)

    return vec

def training_input_base_model(Document, Class):


    kf = KFold(n_splits=10)

    probability1=[]

    for train_index, test_index in kf.split(Document):


        train_reports = np.array(Document)[train_index.astype(int)]
        train_labels = np.array(Class)[train_index.astype(int)]
        test_reports = np.array(Document)[test_index.astype(int)]
        test_labels = np.array(Class)[test_index.astype(int)]

        # Get the prediction probability!!

        prob=get_probability(trainDoc=train_reports, trainClass=train_labels,testDoc=test_reports, testClass=test_labels,tfIdf=True)
        probability1.append(prob)

    return probability1




def get_probability(trainDoc, trainClass,testDoc, testClass,tfIdf):

        vec = tf_idf_func(tfIdf)

        classifier = Pipeline([('vec', vec),
                               ('cls', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                                            metric_params=None,
                                                            n_jobs=1, n_neighbors=31, p=2, weights='uniform'))])

        # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
        classifier.fit(trainDoc, trainClass)

        # Use the classifier to predict the class for all the documents in the test set testDoc
        # Save those output class labels in testGuess
        #testGuess = classifier.predict(testDoc)
        test_Guess_probability = classifier.predict_proba(testDoc)

        return test_Guess_probability















