from pprint import pprint
import matplotlib.pyplot as plt
from read_mnist import *
from classify_mnist import *

if len(sys.argv) != 6:
    print "Please provide valid input parameters ([classifier name] [path to MNIST files] [number of samples]" \
          "[optimize] [cross-validate])"
    exit(1)

# User inputs
clf_name = sys.argv[1]
path = sys.argv[2]
num_samples = int(sys.argv[3])
optimize = bool(int(sys.argv[4]))
cross_validate = bool(int(sys.argv[5]))

# Load training and testing data
print 'Reading the MNIST data...\n'
train_labels, train_images = read_mnist(path, 'training')
test_labels, test_images = read_mnist(path, 'testing')

# Build the MNIST classifier
clf, avg, std = train_classifier(clf_name=clf_name,
                                 labels=train_labels,
                                 examples=train_images,
                                 num_samples=num_samples,
                                 optimize=optimize,
                                 cross_validate=cross_validate)
print 'Hyper-Parameters:'
pprint(clf.get_params())
if optimize:
    print '\nOptimal Hyper-Parameters:'
    pprint(clf.best_params_)
print

if cross_validate:
    print 'Cross-Validation Results:'
    print '  Average:     %f' % (avg)
    print '  Std. Dev.:   %f\n' % (std)

# Test the MNIST classifier
stats = test_classifier(labels=test_labels, examples=test_images, clf=clf)
print 'Testing Results:'
pprint(stats)

# Visualizations
if (clf_name == 'tree' or clf_name == 'forest') and not optimize:
    importances = clf.feature_importances_
    importances = importances.reshape((28, 28))

    # Plot pixel importances
    plt.matshow(importances, cmap=plt.cm.hot)
    plt.title("Pixel importances for %s classifier" % clf_name)
    plt.show()
