from sklearn import svm, tree, ensemble, grid_search, cross_validation
import numpy as np
from pprint import pprint


class HierarchicalRandomForest(object):
    def __init__(self, n_estimators=100, n_procs=1):
        self.classifiers = [ensemble.RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_procs) for i in range(0, 3)]

    def train(self, training_labels, training_examples):
        # Train all three chained random forests
        self.classifiers[0].fit(training_examples, training_labels[:, 0])
        probs = self.classifiers[0].predict_proba(training_examples)

        self.classifiers[1].fit(probs, training_labels[:, 1])
        probs = self.classifiers[1].predict_proba(probs)

        self.classifiers[2].fit(probs, training_labels[:, 2])

    def test(self, test_labels, test_examples):
        # Extract results from the three chained random forest
        results = self.classifiers[0].predict_proba(test_examples)
        results = self.classifiers[1].predict_proba(results)
        results = self.classifiers[2].predict(results)
        
        size = len(test_labels)
        correct_predictions = np.ndarray(size)
        for i in range(size):
            # Record prediction accuracy
            if results[i] == test_labels[i, 2]:
                correct_predictions[i] = True
            else:
                correct_predictions[i] = False

        # Produce statistics dictionary
        stats = dict()
        stats['Prediction Accuracy'] = len([1 for p in correct_predictions if p == True]) / float(size)

        return stats

    def optimize(self):
        raise NotImplementedError, "The optimize() function has yet to be implemented."

    def predict(self, examples):
        # Extract results from the three chained random forest
        results = self.classifiers[0].predict_proba(examples)
        results = self.classifiers[1].predict_proba(results)
        results = self.classifiers[2].predict(results)

        return results
