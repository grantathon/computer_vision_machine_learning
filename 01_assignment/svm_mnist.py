import sys
import numpy as np
from pprint import pprint
from sklearn import svm
from read_mnist import *


def train_mnist_svm(labels, images, num_samples, optimize=False):
    print 'Training the SVM...'
    class_fit = svm.SVC(kernel='poly', degree=2)
    class_fit.fit(images[0:num_samples], labels[0:num_samples])

    return class_fit

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

def optimize_mnist_svm(_svm):
    pass

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Please provide valid input parameters ([path to MNIST files] ['training' or 'testing'] [number of samples])"
        exit(1)
    path = sys.argv[1]
    dataset = sys.argv[2]
    num_samples = int(sys.argv[3])

    # Load training and testing data
    train_labels, train_images = read_mnist(path, 'training')
    test_labels, test_images = read_mnist(path, 'testing')

    # Build the MNIST SVM
    _svm = train_mnist_svm(labels=train_labels, images=train_images, num_samples=num_samples, optimize=False)
    print '\nHyper Parameters:'
    pprint(_svm.get_params())
    print

    # Test the MNIST SVM
    stats = test_mnist_svm(labels=test_labels, images=test_images, _svm=_svm)
    print '\nTesting Results:'
    pprint(stats)
