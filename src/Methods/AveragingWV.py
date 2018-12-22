import logging
#logging.root.handlers = []  # Jupyter messes up logging so needs a reset
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import Avaliar
import gensim
import nltk
import numpy as np
from gensim.models import Word2Vec
from itertools import islice
from nltk import word_tokenize, sent_tokenize
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier



def word_averaging(wv, words):
    all_words = set() 
    mean = []
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:# and wv.syn0norm is not None:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        #logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        #return np.zeros(len(wv.vocab),)
        return np.zeros(wv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list])

def w2v_tokenize_text(text, language, clean):
    tokens = []
    if(language is None):
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if (len(word) < 2 and clean==True):
                    continue
                tokens.append(word)
    else:
        for sent in nltk.sent_tokenize(text, language=language):
            for word in nltk.word_tokenize(sent, language=language):
                if (len(word) < 2 and clean==True):
                    continue
                tokens.append(word)
    return tokens


def classificar(train_data, test_data, my_tags, binary, dirW2V, language, clean, show_confusion_graphic, file_result=None):
    wv = Word2Vec.load_word2vec_format(fname=dirW2V, binary=binary)
    wv.init_sims(replace=True)

    test_tokenized = test_data.apply(lambda r: w2v_tokenize_text(r['plot'], language, clean), axis=1).values
    train_tokenized = train_data.apply(lambda r: w2v_tokenize_text(r['plot'], language, clean), axis=1).values

    X_train_word_average = word_averaging_list(wv,train_tokenized)
    X_test_word_average = word_averaging_list(wv,test_tokenized)
    
    #KNN
    knn_naive_dv = KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine')
    knn_naive_dv.fit(X_train_word_average, train_data.tag)

    predicted1 = knn_naive_dv.predict(X_test_word_average)
    accuracy1 = Avaliar.evaluate_prediction(predicted1, test_data.tag, my_tags, test_data.tag.unique(), show_confusion_graphic, file_result=file_result)


    #Regressao logistica
    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
    logreg = logreg.fit(X_train_word_average, train_data['tag'])
    
    predicted2 = logreg.predict(X_test_word_average)
    accuracy2 = Avaliar.evaluate_prediction(predicted2, test_data.tag, my_tags, test_data.tag.unique(), show_confusion_graphic, file_result=file_result)
    
    #wv.most_similar(positive=[X_test_word_average[56]], restrict_vocab=100000, topn=30)[0:20]
    
    ListPredictions = (predicted1, predicted2)
    ListAccuracy = (accuracy1, accuracy2)
    return ListPredictions, ListAccuracy
