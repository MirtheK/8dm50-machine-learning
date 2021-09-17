import numpy as np
import scipy

def knn_classifier(train_data, train_labels, test_data, k):
    """ k-NN model for classification.
        Returns the labels for test_data, as predicted by the k-NN classifier.
        The label is determined by taking the mode of the labels of k-NN.
    Inputs:
        train_data: data to be trained on (num_train x p matrix)
        train_labels: labels corresponding to the train data (num_train x 1 vector)
        test_data: data for which to perform the knn (num_test x p matrix)
        k: number of neighbours to be used (scalar)
    Outputs:
        Predicted labels for test_data (num_test x 1 vector)
    """    
    D = scipy.spatial.distance.cdist(test_data, train_data, metric='euclidean')
    sort_ix = np.argsort(D, axis=1)
    sort_ix_k = sort_ix[:,:k] # Get the k smallest distances
    predicted_labels = train_labels[sort_ix_k]
    predicted_labels = scipy.stats.mode(predicted_labels, axis=1)[0]

    return predicted_labels

def knn_regressor(train_data, train_targets, test_data, k):
    """ k-NN model for regression.
        Returns the targets for test_data, as predicted by the k-NN regressor.
        The target is determined by taking the mean of the targets of k-NN.
    Inputs:
        train_data: data to be trained on (num_train x p matrix)
        train_labels: targets corresponding to the train data (num_train x 1 vector)
        test_data: data for which to perform the knn (num_test x p matrix)
        k: number of neighbours to be used (scalar)
    Outputs:
        Predicted targets for test_data (num_test x 1 vector)
    """
    D = scipy.spatial.distance.cdist(test_data, train_data, metric='euclidean')
    sort_ix = np.argsort(D, axis=1)
    sort_ix_k = sort_ix[:,:k] # Get the k smallest distances
    predicted_labels = train_targets[sort_ix_k]
    predicted_labels = np.mean(predicted_labels, axis=1)

    return predicted_labels

def get_error_score(y_true, y_predicted):
    """ Error score calculation for classification.
        Checks if the labels correspond and gives a score. 
        An error score of 1 means that everything is falsly predicted.
        An error score of 0 means that everything is correctly predicted.
    Inputs:
        y_true: the true labels (num_datapoints x 1 vector)
        y_predicted: the predicted labels (num_datapoints x 1 vector)
    Outputs:
        error score (scalar)
    """
    correct = [i for i, j in zip(y_true, y_predicted) if i == j]
    return (len(y_true)-len(correct))/len(y_true)

def get_mse(y_true, y_predicted):
    """ Error score calculation for regression.
        Calculates the mean squared error between the true targets and predicted targets. 
        An lower mse is desired.
    Inputs:
        y_true: the true targets (num_datapoints x 1 vector)
        y_predicted: the predicted targets (num_datapoints x 1 vector)
    Outputs:
        mean squared error (scalar)
    """
    mse = np.square(np.subtract(y_true, y_predicted)).mean()
    return mse
