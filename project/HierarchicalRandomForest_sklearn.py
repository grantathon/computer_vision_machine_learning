from sklearn import tree, ensemble
from randomforest import *
import numpy as np
from pprint import pprint


class HierarchicalRandomForest(object):
    def __init__(self, n_estimators=100, n_procs=1):
        self.classifiers = [ensemble.RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_procs) for i in range(0, 3)]

    def train(self, training_labels, training_examples, optimize=False):
        if optimize:
            optimize(training_examples, training_labels)

        # Train all three chained random forests
        self.classifiers[0].fit(training_examples, training_labels[:, 0])
        probs = self.classifiers[0].predict_proba(training_examples)
        hierarchy_inputs = np.concatenate([probs, training_examples], axis=1)
        self.classifiers[1].fit(hierarchy_inputs, training_labels[:, 1])
        probs = self.classifiers[1].predict_proba(hierarchy_inputs)
        hierarchy_inputs = np.concatenate([probs, training_examples], axis=1)
        self.classifiers[2].fit(hierarchy_inputs, training_labels[:, 2])

        # self.classifiers[0].fit(training_examples, training_labels[:, 0])
        # probs = self.classifiers[0].predict_proba(training_examples)
        # self.classifiers[1].fit(probs, training_labels[:, 1])
        # probs = self.classifiers[1].predict_proba(probs)
        # self.classifiers[2].fit(probs, training_labels[:, 2])

        # self.classifiers[0].fit(training_examples, training_labels[:, 0])
        # probs = self.classifiers[0].predict_proba(training_examples)
        # self.classifiers[2].fit(probs, training_labels[:, 2])

        # self.classifiers[1].fit(training_examples, training_labels[:, 1])
        # probs = self.classifiers[1].predict_proba(training_examples)
        # self.classifiers[2].fit(probs, training_labels[:, 2])

        # self.classifiers[2].fit(training_examples, training_labels[:, 2])

    def test(self, test_labels, test_examples, num_classes, top_n=[1, 5]):
        if num_classes < np.max(top_n):
            raise AttributeError, "The number of classes must be greater than or equal to all of the top-n values during testing."

        size = len(test_labels)
        
        # Extract results from the three chained random forest
        probs = self.classifiers[0].predict_proba(test_examples)
        hierarchy_inputs = np.concatenate([probs, test_examples], axis=1)
        probs = self.classifiers[1].predict_proba(hierarchy_inputs)
        hierarchy_inputs = np.concatenate([probs, test_examples], axis=1)
        prediction_probs = self.classifiers[2].predict_proba(hierarchy_inputs)

        # probs = self.classifiers[0].predict_proba(test_examples)
        # probs = self.classifiers[1].predict_proba(probs)
        # prediction_probs = self.classifiers[2].predict_proba(probs)

        # probs = self.classifiers[0].predict_proba(test_examples)
        # prediction_probs = self.classifiers[2].predict_proba(probs)

        # probs = self.classifiers[1].predict_proba(test_examples)
        # prediction_probs = self.classifiers[2].predict_proba(probs)

        # prediction_probs = self.classifiers[2].predict_proba(test_examples)
        
        classes = self.classifiers[2].classes_
        sorted_class_probs = []

        for i in range(size):
            class_probs = [(classes[j], prediction_probs[i][j]) for j in range(num_classes)]
            sorted_class_probs.append(sorted(class_probs, key=lambda tup: tup[1], reverse=True))

        top_n_hits = {}
        for t_n in top_n:
            top_n_hits[t_n] = np.ndarray(size)

        for i in range(size):
            for t_n in top_n_hits.keys():
                top_n = [tup[0] for tup in sorted_class_probs[i][0:t_n]]
                in_top_n = test_labels[i, 2] in top_n

                # Record prediction accuracy
                if in_top_n:
                    top_n_hits[t_n][i] = True
                else:
                    top_n_hits[t_n][i] = False

        # Produce statistics dictionary
        stats = {}
        for key, val in top_n_hits.iteritems():
            stats['top %d accuracy' % key] = len([1 for p in val if p == True]) / float(size)

        return stats

    def optimize(self, training_examples, training_labels):
        raise NotImplementedError, "The optimize() function has yet to be implemented."
        
        # # Initialize hyper-parameter space
        # param_grid = [
        #     {'criterion': ['gini'], 'max_depth': [None, 5, 6, 7, 8, 9, 10], 'n_estimators': [10, 20, 30, 40, 50, 75, 100, 150, 200],
        #      'max_features': [None, int, float, 'auto', 'sqrt', 'log2']},
        #     {'criterion': ['entropy'], 'max_depth': [None, 5, 6, 7, 8, 9, 10], 'n_estimators': [10, 20, 30, 40, 50, 75, 100, 150, 200],
        #      'max_features': [None, int, float, 'auto', 'sqrt', 'log2']}
        # ]

        # # Optimize classifier over hyper-parameter space
        # clf = grid_search.GridSearchCV(estimator=ensemble.RandomForestClassifier(), param_grid=param_grid, scoring='accuracy')
        # self.classifiers[2] = clf

    def predict(self, examples):
        # Extract results from the three chained random forest
        results = self.classifiers[0].predict_proba(examples)
        results = self.classifiers[1].predict_proba(results)
        results = self.classifiers[2]