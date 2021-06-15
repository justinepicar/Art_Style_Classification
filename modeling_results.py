import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pred_results(train_gen, test_gen, pred, csvfile):
    '''
    Create a dataframe with true values and predictions
    :param train_gen: train_generator
    :param test_gen: test_generator
    :param pred: predictions values
    :param csvfile: name of csvfile to import
    :return: results: a dataframe with true values and predicstions for each image
    '''

    pred_class = np.argmax(pred, axis=1)

    labels = (train_gen.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in pred_class]
    true_labels = [labels[k] for k in test_gen.labels]

    filenames = test_gen.filenames
    results = pd.DataFrame({"Filename": filenames,
                            "True_Label": true_labels,
                            "Predictions": predictions})
    results.to_csv(f'../results/{csvfile}.csv', index=False)
    return results


def plot_confusion_matrix(cm):
    '''
    plots confusion matrix
    :param cm: confusion matrix data
    :return: confusion matrix plot
    '''
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.show()