import sys
import numpy as np
from pprint import pprint
from sklearn import svm, grid_search
from read_mnist import *


def train_mnist_svm(labels, images, num_samples, optimize=False):
    if optimize:
        print 'Training the SVM w/optimization...'
        clf = optimize_mnist_svm(opt_method='grid_search')
    else:
        print 'Training the SVM w/o optimization...'
        clf = svm.SVC(kernel='poly', degree=2)

    clf.fit(images[0:num_samples], labels[0:num_samples])

    return clf

def test_mnist_svm(labels, images, _svm):
    size = len(labels)
    correct_predictions = np.ndarray(shape=size)

    print 'Testing the SVM...'
    for i in range(size):
        result = _svm.predict(images[i])

        # Record prediction accuracy
        if result == labels[i]:
            correct_predictions[i] = True
        else:
            correct_predictions[i] = False

    # Produce statistics dictionary
    stats = dict()
    stats['Prediction Accuracy'] = len([1 for p in correct_predictions if p == True]) / float(size)

    return stats

def optimize_mnist_svm(opt_method):
    if opt_method == 'grid_search':
        # Default hyper-parameter grid
        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.0, 0.001, 0.0001], 'kernel': ['rbf']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.0, 0.001, 0.0001], 'kernel': ['sigmoid']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.0, 0.001, 0.0001], 'degree': [2, 3], 'kernel': ['poly']},
        ]

        # Perform an exhaustive search over the hyper-parameter grid
        clf = grid_search.GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, scoring='accuracy')
    else:
        print "ERROR: Unexpected optimization method %s" % (opt_method)
        exit(1)

    return clf

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Please provide valid input parameters ([path to MNIST files] ['training' or 'testing'] [number of samples] [optimize])"
        exit(1)
    path = sys.argv[1]
    dataset = sys.argv[2]
    num_samples = int(sys.argv[3])
    optimize = bool(int(sys.argv[4]))

    # Load training and testing data
    train_labels, train_images = read_mnist(path, 'training')
    test_labels, test_images = read_mnist(path, 'testing')

    # Build the MNIST SVM
    _svm = train_mnist_svm(labels=train_labels, images=train_images, num_samples=num_samples, optimize=optimize)
    print '\nHyper-Parameters:'
    pprint(_svm.get_params())
    if optimize:
        print '\nOptimal Hyper-Parameters:'
        pprint(_svm.best_params_)
    print

    # Test the MNIST SVM
    stats = test_mnist_svm(labels=test_labels, images=test_images, _svm=_svm)
    print '\nTesting Results:'
    pprint(stats)
