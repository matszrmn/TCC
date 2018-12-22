import pandas as pd
import matplotlib.pyplot as plt


def initDataFrame(diretorio, sep, encoding):
    df = pd.read_csv(diretorio, sep=sep, encoding=encoding)
    df = df.dropna() #Retira valores N/A
    df = df.sort_values('tag')
    df['plot'].apply(lambda x: len(x.split(' '))).sum()
    
    df = df.drop('dataID', 1)
    index_column = []
    for i in range(0, len(df)):
        index_column.append(i)
    df.insert(0, 'dataID', index_column)
    df.set_index('dataID', drop=False, inplace=True)
    df['tag'] = [str(i) for i in df['tag']]
    return df

def reset_dataID_column(df):
    df = df.drop('dataID', 1)
    index_column = []
    for i in range(0, len(df)):
        index_column.append(i)
    df.insert(0, 'dataID', index_column)
    df.set_index('dataID', drop=False, inplace=True)
    return df

def plotDataFrameSet(dfs):
    dfs.tag.value_counts().plot(kind="bar", rot=0)
    plt.show()

def printDataFrameCell(df, index):
    example = df[df.index == index][['plot', 'tag']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Genre:', example[1])

def k_folds(k, dataset):
    if(len(dataset) < 1): return None

    my_tags = dataset.tag.unique()
    
    begin_indexes = list() #Indice da primeira instancia de cada "tag"
    begin_indexes.append(0)
    index_tag_atual = 0
    
    for i in range (1, len(dataset)):
        if(dataset['tag'][i] != my_tags[index_tag_atual]):
            begin_indexes.append(i)
            index_tag_atual += 1
            #if(index_tag_atual >= len(my_tags)): break

    total_instancias = [0] * len(my_tags) #Quantidade total de instancias de cada "tag"

    for current_tag in range (0, len(my_tags)):
        if(current_tag != len(my_tags)-1): total_instancias[current_tag] = begin_indexes[current_tag+1] - begin_indexes[current_tag]
        else:                              total_instancias[current_tag] = len(dataset) - begin_indexes[current_tag]

    folds = list()
    index_inicio = 0
    index_fim = 0

    quant_instancias = 0
    deslocamento = 0

    for current_fold in range (0, k):
        index_fold_atual = 0
        frame = pd.DataFrame(columns=['dataID','plot','tag'])

        for current_tag in range (0, len(my_tags)):
            quant_instancias = int(total_instancias[current_tag]/k)
            deslocamento = quant_instancias * current_fold
            index_inicio = begin_indexes[current_tag] + deslocamento
            
            if(current_fold != k-1):
                index_fim = index_inicio + quant_instancias
            else: 
                if(current_tag != len(my_tags)-1): index_fim = begin_indexes[current_tag+1]
                else: index_fim = len(dataset)
            
            for i in range (index_inicio, index_fim):
                frame.loc[index_fold_atual] = dataset.iloc[i]
                index_fold_atual += 1

        frame.set_index('dataID',drop=False)
        frame = reset_dataID_column(frame)
        folds.append(frame)

    return folds        

