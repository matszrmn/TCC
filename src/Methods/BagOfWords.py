#import logging
#logging.root.handlers = []  # Jupyter messes up logging so needs a reset
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import Avaliar
import nltk
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer



#def most_influential_words(vectorizer, genre_index=0, num_words=10):
#    features = vectorizer.get_feature_names()
#    max_coef = sorted(enumerate(logreg.coef_[genre_index]), key=lambda x:x[1], reverse=True)
#    return [features[x[0]] for x in max_coef[:num_words]]  

def classificar(train_data, test_data, my_tags, language, max_features, show_confusion_graphic, file_result=None):
    count_vectorizer = CountVectorizer(
        analyzer="word", tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words=language, max_features=max_features) 
    train_data_features = count_vectorizer.fit_transform(train_data['plot'])

    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
    logreg = logreg.fit(train_data_features, train_data['tag'])
    
    #joblib.dump(logreg, 'filename.pkl') #PERSISTENCIA (escrita)
    #logreg = joblib.load('filename.pkl') #PERSISTENCIA (leitura)
    
    prediction, accuracy = Avaliar.predict(count_vectorizer, logreg, test_data, my_tags, test_data.tag.unique(), show_confusion_graphic, file_result=file_result)
    return prediction, accuracy


    #count_vectorizer.get_feature_names()[80:90]
    #genre_tag_id = 1
    #print(my_tags[genre_tag_id])
    #most_influential_words(count_vectorizer, genre_tag_id)

