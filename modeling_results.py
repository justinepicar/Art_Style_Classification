import pandas as pd
import matplotlib.pyplot as plt

def pred_results(train_gen, test_gen, pred, csvfile):

    pred_class = np.argmax(pred, axis=1)

    labels = (train_gen.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in pred_class]
    true_labels = [labels[k] for k in test_gen.labels]

    filenames = test_gen.filenames
    results = pd.DataFrame({"Filename": filenames,
                            "True_Label": true_labels,
                            "Predictions": predictions})
    results.to_csv(f'{csvfile}.csv', index=False)
    return results


def plot_confusion_matrix(cm):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.show()