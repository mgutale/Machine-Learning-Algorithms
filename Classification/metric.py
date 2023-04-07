def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of a classification model.

    Parameters:
        y_true (array-like): True labels of the data.
        y_pred (array-like): Predicted labels of the data.

    Returns:
        float: Accuracy of the model as a percentage (between 0 and 1).
    """
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix of a classification model.

    Parameters:
        y_true (array-like): True labels of the data.
        y_pred (array-like): Predicted labels of the data.

    Returns:
        array: Confusion matrix, where the rows represent the true labels and the columns represent the predicted labels.
    """
    labels = np.unique(y_true)
    n_labels = len(labels)
    matrix = np.zeros((n_labels, n_labels), dtype=int)
    for i in range(n_labels):
        for j in range(n_labels):
            matrix[i, j] = np.sum((y_true == labels[i]) & (y_pred == labels[j]))
    return matrix
