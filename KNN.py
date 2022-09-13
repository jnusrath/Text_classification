import Eval_Matrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

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

def KNN_prob(trainDoc, trainClass, testDoc, testClass, tfIdf):



    vec = tf_idf_func(tfIdf)


    classifier = Pipeline([('vec', vec),
                           ('cls', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
                                   n_jobs=1, n_neighbors=31, p=2, weights='uniform'))])

    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc, trainClass)


    test_Guess_probability=classifier.predict_proba(testDoc)

    return test_Guess_probability



