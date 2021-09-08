import numpy as np
import scipy

def knn_classifier(train_data, train_labels, test_data, k):
    # Returns the labels for test_data, predicted by the k-NN
    # clasifier trained on train_data and train_labels
    # Input:
    # train_data - num_train x p matrix with features for the training data
    # train_labels - num_train x 1 vector with labels for the training data
    # test_data - num_test x p matrix with features for the test data
    # k - Number of neighbors to take into account (1 by default)
    # Output:
    # predicted_labels - num_test x 1 predicted vector with labels for the test data
    
    D = scipy.spatial.distance.cdist(test_data, train_data, metric='euclidean')
    sort_ix = np.argsort(D, axis=1)
    sort_ix_k = sort_ix[:,:k] # Get the k smallest distances
    predicted_labels = train_labels[sort_ix_k]
    predicted_labels = scipy.stats.mode(predicted_labels, axis=1)[0]

    return predicted_labels

def get_error_score(y_true, y_predicted):
    correct = [i for i, j in zip(y_true, y_predicted) if i == j]
    return (len(y_true)-len(correct))/len(y_true)
