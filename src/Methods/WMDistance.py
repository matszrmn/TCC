import logging
logging.root.handlers = []  # Jupyter messes up logging so needs a reset
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import Avaliar
import nltk
import numpy as np
from gensim.models import KeyedVectors
#from gensim.models import Word2Vec
import pandas as pd
from pyemd import emd
from nltk.corpus import stopwords
from sklearn.metrics import euclidean_distances
from sklearn.externals.joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array
from sklearn.cross_validation import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer



class WordMoversKNN(KNeighborsClassifier):
    _pairwise = False

    def __init__(self, W_embed, n_neighbors=1, n_jobs=1, verbose=False):
        self.W_embed = W_embed
        self.verbose = verbose
        super(WordMoversKNN, self).__init__(n_neighbors=n_neighbors, n_jobs=n_jobs,
                                            metric='precomputed', algorithm='brute')

    def _wmd(self, i, row, X_train):
        union_idx = np.union1d(X_train[i].indices, row.indices)
        W_minimal = self.W_embed[union_idx]
        W_dist = euclidean_distances(W_minimal)
        bow_i = X_train[i, union_idx].A.ravel()
        bow_j = row[:, union_idx].A.ravel()
        return emd(bow_i, bow_j, W_dist)
    
    def _wmd_row(self, row, X_train):
        n_samples_train = X_train.shape[0]
        return [self._wmd(i, row, X_train) for i in range(n_samples_train)]

    def _pairwise_wmd(self, X_test, X_train=None):
        n_samples_test = X_test.shape[0]
        
        if X_train is None:
            X_train = self._fit_X

        dist = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._wmd_row)(test_sample, X_train)
            for test_sample in X_test)

        return np.array(dist)

    def fit(self, X, y):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        return super(WordMoversKNN, self).fit(X, y)

    def predict(self, X):
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        return super(WordMoversKNN, self).predict(dist)
    
    
class WordMoversKNNCV(WordMoversKNN):
    def __init__(self, W_embed, n_neighbors_try=None, scoring=None, cv=3,
                 n_jobs=1, verbose=False):
        self.cv = cv
        self.n_neighbors_try = n_neighbors_try
        self.scoring = scoring
        super(WordMoversKNNCV, self).__init__(W_embed,
                                              n_neighbors=None,
                                              n_jobs=n_jobs,
                                              verbose=verbose)

    def fit(self, X, y):
        if self.n_neighbors_try is None:
            n_neighbors_try = range(1, 6)
        else:
            n_neighbors_try = self.n_neighbors_try

        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)

        cv = check_cv(self.cv, X, y)
        knn = KNeighborsClassifier(metric='precomputed', algorithm='brute')
        scorer = check_scoring(knn, scoring=self.scoring)

        scores = []
        for train_ix, test_ix in cv:
            dist = self._pairwise_wmd(X[test_ix], X[train_ix])
            knn.fit(X[train_ix], y[train_ix])
            scores.append([
                scorer(knn.set_params(n_neighbors=k), dist, y[test_ix])
                for k in n_neighbors_try
            ])
        scores = np.array(scores)
        self.cv_scores_ = scores

        best_k_ix = np.argmax(np.mean(scores, axis=0))
        best_k = n_neighbors_try[best_k_ix]
        self.n_neighbors = self.n_neighbors_ = best_k

        return super(WordMoversKNNCV, self).fit(X, y)

def w2v_tokenize_text(text, language):
    tokens = []
    for sent in nltk.sent_tokenize(text, language=language):
        for word in nltk.word_tokenize(sent, language=language):
            if len(word) < 2:
                continue
            if word in stopwords.words(language):
                continue
            tokens.append(word)
    return tokens

def concatenate_tokens(tokens):
    string = ""
    if(not tokens): return string

    string = string + tokens[0]
    for i in range(1,len(tokens)):
        string = string + " " + tokens[i]
    
    return string


def classificar(train_data, test_data, my_tags, dirW2V, binary, dirData, shapeLin, shapeCol, language, show_confusion_graphic, file_result=None):
    
    train_tokenized = []
    current_tokens = []
    train_new_data = pd.DataFrame(columns=('plot','tag'))
    
    for i in range(0,len(train_data)):
        current_tokens = w2v_tokenize_text(train_data['plot'][i], language)
        train_tokenized.append(current_tokens)
        train_new_data.loc[i] = [concatenate_tokens(current_tokens), train_data['tag'][i]]
    
    #train_new_data['dataID'] = train_new_data.index
    del train_data
    

    #test_tokenized = []
    test_new_data = pd.DataFrame(columns=('plot','tag'))

    for i in range(0,len(test_data)):
        current_tokens = w2v_tokenize_text(test_data['plot'][i], language)
        #test_tokenized.append(current_tokens)
        test_new_data.loc[i] = [concatenate_tokens(current_tokens), test_data['tag'][i]]

    #test_new_data['dataID'] = test_new_data.index
    del test_data
    del current_tokens



    wv = KeyedVectors.load_word2vec_format(dirW2V, binary=binary)
    wv.init_sims(replace=True)

    fp = np.memmap(dirData + "embed.dat", dtype=np.double, mode='w+', shape=wv.syn0norm.shape) 
    fp[:] = wv.syn0norm[:]

    with open(dirData + "embed.vocab", "w") as f:
        for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
            print(w, file=f)
    del fp, wv

    W = np.memmap(dirData + "embed.dat", dtype=np.double, mode="r", shape=(shapeLin, shapeCol))
    with open(dirData + "embed.vocab") as f:
        vocab_list = map(str.strip, f.readlines())

    vocab_dict = {w: k for k, w in enumerate(vocab_list)}



    flat_train_tokenized = [item for sublist in train_tokenized for item in sublist]
    del train_tokenized #talvez retirar

    vect = CountVectorizer(stop_words=language).fit(flat_train_tokenized)
    del flat_train_tokenized #talvez retirar
    
    common = [word for word in vect.get_feature_names() if word in vocab_dict]
    W_common = W[[vocab_dict[w] for w in common]]

    del W #talvez retirar
    del vocab_dict #talvez retirar


    vect = CountVectorizer(vocabulary=common, dtype=np.double)
    del common #talvez retirar

    X_train = vect.fit_transform(train_new_data['plot'])
    
    X_test = vect.transform(test_new_data['plot']) #test_data['plot']
    del vect #talvez retirar
    #del test_tokenized #talvez retirar

    knn = WordMoversKNN(n_neighbors=1, W_embed=W_common, verbose=5, n_jobs=7)
    del W_common #talvez retirar

    knn.fit(X_train, train_new_data['tag'])
    del train_new_data
    del X_train #talvez retirar

    prediction = knn.predict(X_test)
    del X_test #talvez retirar
    del knn #talvez retirar
 
    accuracy = Avaliar.evaluate_prediction(prediction, test_new_data.tag, my_tags, test_new_data.tag.unique(), show_confusion_graphic, file_result=file_result)
    return prediction, accuracy
