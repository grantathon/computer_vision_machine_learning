from read_mnist import *
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

path = '../data'
num_sample = -1

train_labels, train_images = read_mnist(path, 'training')
test_labels, test_images = read_mnist(path, 'testing')
if num_sample < 0:
        num_sample = len(train_labels)

classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features = None)
classifier.fit(train_images[:num_sample], train_labels[:num_sample])

#dot_data = StringIO() 
#tree.export_graphviz(classifier, out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("tree.pdf") 

with open("tree.dot", 'w') as f:
	f = tree.export_graphviz(classifier, out_file=f)