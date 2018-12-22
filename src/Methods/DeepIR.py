import logging
logging.disable(logging.INFO)
#logging.root.handlers = []  # Jupyter messes up logging so needs a reset
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level='ERROR')

import Avaliar
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import re
from copy import deepcopy
from gensim.models import Word2Vec
from numpy import random
from sklearn.externals import joblib



def clean(text): #Deixar texto no formato adequado
    contractions = re.compile(r"'|-|\"")
    symbols = re.compile(r'(\W+)', re.U) 		#Removendo todos os nao alfa-numericos
    singles = re.compile(r'(\s\S\s)', re.I|re.U) 	#Removendo caracteres sozinhos
    seps = re.compile(r'\s+') 			#Removendo separadores
    
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

def sentences(l): #Divisor de frases
    alteos = re.compile(r'([!\?])')
    l = alteos.sub(r' \1 .', l).rstrip("(\.)*\n")
    return l.split(".")

def plots(data, clean_text): #Limpar train_data ou test_data
    my_df = data
    for i, row in my_df.iterrows():
        if(clean_text==True):
            yield {'y':row['tag'],\
            'x':[clean(s).split() for s in sentences(row['plot'])]}
        else:
            yield {'y':row['tag'],\
            'x':[s.split() for s in sentences(row['plot'])]}

def docprob(docs, mods):
    sentlist = [s for d in docs for s in d] #Lista de frases
    llhd = np.array( [ m.score(sentlist, len(sentlist)) for m in mods ] ) #Grau de proximidade entre cada frase
    lhd = np.exp(llhd - llhd.max(axis=0)) #Calculando grau de proximidade atraves da potencia e subtraindo de max para evitar sobrecarga
    prob = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose()) #Normalizacao para obter probabilidades
    prob["doc"] = [i for i,d in enumerate(docs) for s in d] #Media de probabilidades das frases para analisar
    prob = prob.groupby("doc").mean()
    return prob

def print_box_plot(revtest, probs, my_tags):
    tag_index = 0
    col_name = "out-of-sample prob positive for " + my_tags[tag_index]
    probpos = pd.DataFrame({col_name:probs[[tag_index]].sum(axis=1), 
                           "true genres": [r['y'] for r in revtest]})
    probpos.boxplot(col_name,by="true genres", figsize=(12,5))
    plt.show()


#Metodo principal
def classificar(train_data, test_data, my_tags, number_iters, size, alpha, clean, show_box_plot, show_confusion_graphic, file_result=None):
    def tag_sentences(reviews, stars=my_tags):  
        for r in reviews:
            if r['y'] in stars:
                for s in r['x']:
                    yield s

    revtrain = list(plots(train_data, clean))
    revtest = list(plots(test_data, clean))

    np.random.shuffle(revtrain)
    #print(next(tag_sentences(revtrain, my_tags[0])))
    
    basemodel = Word2Vec(workers=multiprocessing.cpu_count(), #Processamento em paralelo
                         iter=number_iters, #Tempo de aprendizado
                         hs=1, negative=0, #Apenas classificacao para tipo "softmax" 
                         )
    #print(basemodel)
    basemodel.build_vocab(tag_sentences(revtrain))
    genremodels = [deepcopy(basemodel) for i in range(len(my_tags))]
    for i in range(len(my_tags)):
        slist = list(tag_sentences(revtrain, my_tags[i]))
        genremodels[i].train(  slist, total_examples=len(slist) )


    Word2Vec(size=size, alpha=alpha) #Incluir argumento vocab=0 se nao houver erro

    #joblib.dump(genremodels,'filename.pkl') #PERSISTENCIA (escrita)
    #genremodels = joblib.load('filename.pkl') #PERSISTENCIA (leitura)


    #Prevendo dados
    probs = docprob( [r['x'] for r in revtest], genremodels)  
    prediction = probs.idxmax(axis=1).apply(lambda x: my_tags[x])

    if(show_box_plot==True):
        print_box_plot(revtest, probs, my_tags)        

    target = [r['y'] for r in revtest]
    accuracy = Avaliar.evaluate_prediction(prediction, target, my_tags, test_data.tag.unique(), show_confusion_graphic, file_result=file_result)
    return prediction, accuracy
    
