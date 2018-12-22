import Inicializar
import Avaliar
import numpy as np
import pandas as pd
from Methods import BagOfWords
from Methods import CharacterNgrams
from Methods import TfIdf
from Methods import DeepIR
from Methods import WMDistance
from Methods import AveragingWV
from Methods import Doc2Vec
from sklearn.model_selection import train_test_split
print('\n\n\n')


dirW2V = '/home/pc/Data/WMD/GoogleNews-vectors-negative300.bin.gz'
dirData = '/home/pc/Data/WMD/Data'



def executar_sem_validacao(method, train_data, test_data, my_tags, clean, show_box_plot, show_confusion_graphic, language, file_result=None):
    prediction = None
    ListPredictions = None
    accuracy = 0
    ListAccuracy = None

    if(method==1):
        max_features = 3000
        prediction, accuracy = BagOfWords.classificar(train_data, test_data, my_tags, language, max_features, show_confusion_graphic,
                                                      file_result=file_result)
    elif(method==2):
        max_features = 3000
        prediction, accuracy = CharacterNgrams.classificar(train_data, test_data, my_tags, max_features, show_confusion_graphic,
                                                           file_result=file_result)
    elif(method==3):
        prediction, accuracy = TfIdf.classificar(train_data, test_data, my_tags, language, show_confusion_graphic,
                                                 file_result=file_result)
    elif(method==4):
        number_iters = 100
        size = 100
        alpha = 0.025
        prediction, accuracy = DeepIR.classificar(train_data, test_data, my_tags, number_iters, size, alpha, clean, show_box_plot, 	                                            show_confusion_graphic, file_result=file_result)
    elif(method==5):
        ListPredictions, ListAccuracy = Doc2Vec.classificar(train_data, test_data, my_tags, 20, clean, show_confusion_graphic,
                                                            file_result=file_result)
    elif(method==6):
        binary = True
        ListPredictions, ListAccuracy = AveragingWV.classificar(train_data, test_data, my_tags, binary, dirW2V, 
                                                                language, clean, show_confusion_graphic, file_result=file_result)
    else:
        binary = True
        prediction , accuracy = WMDistance.classificar(train_data, test_data, my_tags, dirW2V, binary, dirData, 
                                                      3000000, 300, language, show_confusion_graphic, file_result=file_result)
    #if(ListAccuracy is None):
    #    print('Final accuracy = %s' %str(accuracy))
    #    file_result.write('Final accuracy = %s\n' %str(accuracy))
    #else:
    #    print('Final accuracy = %s' %str(ListAccuracy))
    #    file_result.write('Final accuracy = %s\n' %str(ListAccuracy))
    #    file_result.write('\n')


def executar_com_validacao(method, folds, my_tags, clean, show_box_plot, show_confusion_graphic, language, file_result=None):
    train_data = None
    prediction = None
    ListPredictions = None
    
    accuracy = 0
    ListAccuracy = None
    cumulated_accuracy = 0
    cumulated_list_accuracy = None

    for i in range(0, len(folds)):
        print('(Etapa %d de %d)' % (i+1, len(folds)))
        file_result.write('(Etapa %d de %d)\n' % (i+1, len(folds)))

        for j in range(0, len(folds)):
            if(j != i):
                if(train_data is None): train_data = folds[j].copy()
                else: train_data = pd.concat([train_data, folds[j]])

        if(method==1):
            max_features = 3000
            prediction, accuracy = BagOfWords.classificar(train_data, folds[i], my_tags, language, max_features, show_confusion_graphic,
                                                          file_result=file_result)
        elif(method==2):
            max_features = 3000
            prediction, accuracy = CharacterNgrams.classificar(train_data, folds[i], my_tags, max_features, show_confusion_graphic,
                                                               file_result=file_result)
        elif(method==3):
            prediction, accuracy = TfIdf.classificar(train_data, folds[i], my_tags, language, show_confusion_graphic,
                                                     file_result=file_result)
        elif(method==4):
            number_iters = 100
            size = 100
            alpha = 0.025
            prediction, accuracy = DeepIR.classificar(train_data, folds[i], my_tags, number_iters, size, alpha, clean, show_box_plot, 	                                                    show_confusion_graphic, file_result=file_result)
        elif(method==5):
            ListPredictions, ListAccuracy = Doc2Vec.classificar(train_data, folds[i], my_tags, 20, clean, show_confusion_graphic,
                                                                file_result=file_result)
            if(cumulated_list_accuracy is None): cumulated_list_accuracy = np.asarray(ListAccuracy)
            else: 
                for k in range(0, len(ListAccuracy)): cumulated_list_accuracy[k] = cumulated_list_accuracy[k] + ListAccuracy[k]
        elif(method==6):
            binary = True
            ListPredictions, ListAccuracy = AveragingWV.classificar(train_data, folds[i], my_tags, binary, dirW2V, 
                                                                    language, clean, show_confusion_graphic, file_result=file_result)
            if(cumulated_list_accuracy is None): cumulated_list_accuracy = np.asarray(ListAccuracy)
            else: 
                for k in range(0, len(ListAccuracy)): cumulated_list_accuracy[k] = cumulated_list_accuracy[k] + ListAccuracy[k]
        else:
            binary = True
            prediction , accuracy= WMDistance.classificar(train_data, folds[i], my_tags, dirW2V, binary, dirData, 
                                                          3000000, 300, language, show_confusion_graphic, file_result=file_result)
        cumulated_accuracy += accuracy
        train_data = None
    
    if(cumulated_list_accuracy is None):
        print('Final accuracy = %s' %str(cumulated_accuracy/len(folds)))
        file_result.write('Final accuracy = %s\n' %str(cumulated_accuracy/len(folds)))
    else:
        for k in range(0, len(cumulated_list_accuracy)): cumulated_list_accuracy[k] = cumulated_list_accuracy[k]/len(folds)
        print('Final accuracy = %s' %str(cumulated_list_accuracy))
        file_result.write('Final accuracy = %s\n' %str(cumulated_list_accuracy))


#Inicializando e executando
###################################################################################################
#Obs: metodo 7 pode alterar estrutura de treino e de teste


#Com validacao cruzada K-FOLDS
df = Inicializar.initDataFrame('/home/tcc2/testando2/tcc_6/data/age.csv', '\t', 'utf8')
my_tags = df.tag.unique()
folds = Inicializar.k_folds(10, df)
del(df)
file_result = open('results.txt','a')
executar_com_validacao(1, folds, my_tags, True, False, False, 'english', file_result)
file_result.close()

#Com validacao cruzada HOLDOUT
#df = Inicializar.initDataFrame('/home/tcc2/testando/tagged_plots_movielens.csv', ',', 'latin1')
#my_tags = df.tag.unique()
#train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
#train_data = Inicializar.reset_dataID_column(train_data)
#test_data = Inicializar.reset_dataID_column(test_data)
#del(df)
#file_result = open('results.txt','a')
#executar_sem_validacao(7, train_data, test_data, my_tags, True, False, False, 'english', file_result)
#file_result.close()

#Sem validacao cruzada:
#train_data = Inicializar.initDataFrame('/home/pc/Data/Entradas/Train/Train_En_Age.csv', '\t', 'latin1') #latin1 ou utf8
#test_data = Inicializar.initDataFrame('/home/pc/Data/Entradas/Test1/Test1_En_Age.csv', '\t', 'latin1') #latin1 ou utf8
#my_tags = train_data.tag.unique()
#file_result = open('results.txt','a')
#executar_sem_validacao(7, train_data, test_data, my_tags, True, False, False, 'english', file_result)
#file_result.close()

