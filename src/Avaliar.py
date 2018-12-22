import matplotlib.pyplot as plt
import decimal
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def plot_confusion_matrix(cm, test_data_tags, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(test_data_tags))
    target_names = test_data_tags
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate_prediction(predictions, target, my_tags, test_data_tags, show_confusion_graphic, title="Confusion matrix", file_result=None):
    accuracy = accuracy_score(target, predictions)
    cm_matrix = confusion_matrix(target, predictions)
    
    if(len(cm_matrix) == len(my_tags)): test_data_tags = my_tags

    cm_df = pd.DataFrame(cm_matrix, index=test_data_tags, columns=test_data_tags) #ERRO   
    print('accuracy %s \n' % accuracy)
    print('%s' % cm_df)
    print('(row=expected, col=predicted)\n')
    
    if(file_result is not None):
        file_result.write('accuracy %s \n\n' % accuracy)
        file_result.write('%s\n' % cm_df)
        file_result.write('(row=expected, col=predicted)\n\n')

    if(show_confusion_graphic):
        cm_normalized = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(cm_normalized, test_data_tags)
        plt.show()
    return accuracy

def predict(vectorizer, classifier, data, my_tags, test_data_tags, show_confusion_graphic, file_result=None):
    data_features = vectorizer.transform(data['plot'])
    predictions = classifier.predict(data_features)
    target = data['tag']
    accuracy = evaluate_prediction(predictions, target, my_tags, test_data_tags, show_confusion_graphic, file_result=file_result)
    return predictions, accuracy
