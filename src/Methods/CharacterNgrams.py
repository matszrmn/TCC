#import logging
#logging.root.handlers = []  # Jupyter messes up logging so needs a reset
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import Avaliar
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer



def classificar(train_data, test_data, my_tags, max_features, show_confusion_graphic, file_result=None):
    n_gram_vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=([2,5]),
        tokenizer=None,    
        preprocessor=None,                               
        max_features=max_features) 

    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
    train_data_features = n_gram_vectorizer.fit_transform(train_data['plot'])
    logreg = logreg.fit(train_data_features, train_data['tag'])

    #joblib.dump(logreg, 'filename.pkl') #PERSISTENCIA (escrita)
    #logreg = joblib.load('filename.pkl') #PERSISTENCIA (leitura)

    prediction, accuracy = Avaliar.predict(n_gram_vectorizer, logreg, test_data, my_tags, test_data.tag.unique(), show_confusion_graphic, file_result=file_result)
    return prediction, accuracy

    #n_gram_vectorizer.get_feature_names()[50:60]
