from pprint import pprint
from read_mnist import *
from learn_mnist import *

if len(sys.argv) != 5:
    print "Please provide valid input parameters ([classifier name] [path to MNIST files] [number of samples] [optimize])"
    exit(1)

# User inputs
clf_name = sys.argv[1]
path = sys.argv[2]
num_samples = int(sys.argv[3])
optimize = bool(int(sys.argv[4]))

# Load training and testing data
train_labels, train_images = read_mnist(path, 'training')
test_labels, test_images = read_mnist(path, 'testing')

# Build the MNIST SVM
clf = train_classifier(clf_name=clf_name,
                       labels=train_labels,
                       examples=train_images,
                       num_samples=num_samples,
                       optimize=optimize)
print '\nHyper-Parameters:'
pprint(clf.get_params())
if optimize:
    print '\nOptimal Hyper-Parameters:'
    pprint(clf.best_params_)
print

# Test the MNIST SVM
stats = test_classifier(labels=test_labels, examples=test_images, clf=clf)
print '\nTesting Results:'
pprint(stats)
