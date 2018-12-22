#import logging
#logging.root.handlers = []  # Jupyter messes up logging so needs a reset
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import Avaliar
import nltk
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer



def classificar(train_data, test_data, my_tags, language, show_confusion_graphic, file_result=None):
    tf_vect = TfidfVectorizer(
        min_df=2, tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words=language)
    train_data_features = tf_vect.fit_transform(train_data['plot'])

    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
    logreg = logreg.fit(train_data_features, train_data['tag'])

    #joblib.dump(logreg, 'filename.pkl') #PERSISTENCIA (escrita)
    #logreg = joblib.load('filename.pkl') #PERSISTENCIA (leitura)

    prediction, accuracy = Avaliar.predict(tf_vect, logreg, test_data, my_tags, test_data.tag.unique(), show_confusion_graphic, file_result=file_result)
    return prediction, accuracy


    #tf_vect.get_feature_names()[1000:1010]
