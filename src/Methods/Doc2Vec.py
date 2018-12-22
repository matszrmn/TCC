import logging
logging.disable(logging.WARNING)
#logging.root.handlers = []  # Jupyter messes up logging so needs a reset
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import Avaliar
import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from numpy import random
from sklearn import linear_model
from sklearn.externals import joblib



def tokenize_text(text, clean):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if (len(word) < 2 and clean==True):
                continue
            tokens.append(word.lower())
    return tokens

def classificar(train_data, test_data, my_tags, seed, clean, show_confusion_graphic, file_result=None):
    train_tagged = train_data.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['plot'], clean), tags=[r.tag]), axis=1)

    test_tagged = test_data.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['plot'], clean), tags=[r.tag]), axis=1)

    trainsent = train_tagged.values
    testsent = test_tagged.values
    doc2vec_model = Doc2Vec(trainsent, workers=1, size=5, iter=20, dm=1)

    train_targets, train_regressors = zip(
        *[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in trainsent])

    test_targets, test_regressors = zip(
        *[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in testsent])
    
    #joblib.dump(doc2vec_model, 'filename.pkl') #PERSISTENCIA (escrita)
    #doc2vec_model = joblib.load('filename.pkl') #PERSISTENCIA (leitura)

    #Knn
    prediction1 = [
        doc2vec_model.docvecs.most_similar([pred_vec], topn=1)[0][0]
        for pred_vec in test_regressors
    ]
    accuracy1 = Avaliar.evaluate_prediction(prediction1, test_targets, my_tags, test_data.tag.unique(), show_confusion_graphic, title=str(doc2vec_model),
                                            file_result=file_result)
    
    #doc2vec_model.docvecs.most_similar('action')
    #doc2vec_model.most_similar([doc2vec_model.docvecs['sci-fi']])
    
    
    #Regressao logistica 1
    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
    logreg = logreg.fit(train_regressors, train_targets)
    #joblib.dump(logreg, 'filename.pkl') #PERSISTENCIA (escrita)
    #logreg = joblib.load('filename.pkl') #PERSISTENCIA (leitura)
    prediction2 = logreg.predict(test_regressors)
    accuracy2 = Avaliar.evaluate_prediction(prediction2, test_targets, my_tags, test_data.tag.unique(), show_confusion_graphic, title=str(doc2vec_model),
                                            file_result=file_result)


    #Regressao logistica 2
    doc2vec_model.seed = seed
    doc2vec_model.random = random.RandomState(seed)
    test_targets, test_regressors = zip(
        *[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in testsent])

    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5, random_state=42)
    logreg = logreg.fit(train_regressors, train_targets)
    #joblib.dump(logreg, 'filename.pkl') #PERSISTENCIA (escrita)
    #logreg = joblib.load('filename.pkl') #PERSISTENCIA (leitura)
    prediction3 = logreg.predict(test_regressors)
    accuracy3 = Avaliar.evaluate_prediction(prediction3, test_targets, my_tags, test_data.tag.unique(), show_confusion_graphic, title=str(doc2vec_model),
                                            file_result=file_result)

    
    ListPredictions = (prediction1, prediction2, prediction3)
    ListAccuracy = (accuracy1, accuracy2, accuracy3)
    return ListPredictions, ListAccuracy
