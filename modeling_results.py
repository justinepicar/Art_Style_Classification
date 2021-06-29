import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def pred_results(train_gen, test_gen, pred, csv_file):
    """
    Create a dataframe with true values and predictions
    :param train_gen: train_generator
    :param test_gen: test_generator
    :param pred: predictions values
    :param csv_file: name of csv file to import
    :return: results: a dataframe with true values and predictions for each image
    """

    pred_class = np.argmax(pred, axis=1)

    labels = train_gen.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in pred_class]
    true_labels = [labels[k] for k in test_gen.labels]

    filenames = test_gen.filenames
    results = pd.DataFrame({"Filename": filenames,
                            "True_Label": true_labels,
                            "Predictions": predictions})
    results.to_csv(f'../results/{csv_file}.csv', index=False)
    return results


def plot_confusion_matrix(cm, labels):
    """
    Displays confusion matrix plot
    :param cm: confusion matrix data
    :param labels: class labels
    :return: None
    """
    df_cm = pd.DataFrame(cm / np.sum(cm), index=[i for i in labels],
                         columns=[i for i in labels])

    plt.figure(figsize=(10, 5))
    sns.heatmap(df_cm, annot=True, cmap='Blues')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title('Confusion matrix')

    return None


def plot_clf(clf):
    """
    Displays classification report heatmap
    :param clf: classification report data
    :return: None
    """

    plt.figure(figsize=(10, 5))
    sns.heatmap(clf.iloc[:, :-1], annot=True)
    plt.title('Classification Report')

    return None


def metric_eval(test_generator, pred, results, labels):
    """
    displays a confusion matrix and classification heatmap
    and returns a classification report dataframe
    :param test_generator: image generator from test set
    :param pred: predicted labels dummy variables
    :param results: dataframe of true and predicted values
    :param labels: class labels
    :return: classification report dataframe
    """
    cm = confusion_matrix(list(results.True_Label), list(results.Predictions), labels=labels)
    plot_confusion_matrix(cm, labels)
    cf = cf_report(test_generator, pred, labels)
    plot_clf(cf)

    return cf


def cf_report(test_generator, pred, labels):
    """
    creates classification report data frame
    :param test_generator: image generator from test set
    :param pred: predicted labels dummy variables
    :param labels: class labels
    :return: classification report dataframe
    """
    predicted_classes = np.argmax(pred, axis=1)
    cf = pd.DataFrame(classification_report(test_generator.classes,
                                            predicted_classes,
                                            target_names=labels,
                                            output_dict=True)).T
    return cf
