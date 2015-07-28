from randomforest import *
import numpy as np
import operator
from pprint import pprint


class HierarchicalRandomForest(object):
    def __init__(self, n_estimators=100, n_procs=1):
        # TODO: Try different parameters
        # params = {  'max_depth' : 10,
        #             'min_sample_count' : 5,
        #             'test_count' : 100,
        #             'test_class' : getattr( weakLearner, learner)()
        #          }

        self.forests = [Forest(ntrees=n_estimators, nprocs=n_procs) for i in range(0, 3)]

    def train(self, training_labels, training_examples, num_classes, optimize=False):
        size = len(training_examples)

        # Train first RF hierarchy
        self.forests[0].grow(points=training_examples, responses=training_labels[:, 0])

        # Extract distribution from all leaves of first RF hierarchy
        prediction_probs = np.ndarray((size, 2))
        for i in range(size):
            prediction_probs[i] = self.forests[0].predict(point=training_examples[i], soft=True).values()

        # Train second RF hierarchy
        self.forests[1].grow(points=prediction_probs, responses=training_labels[:, 1])

        # Extract distribution from all leaves of second RF hierarchy
        temp_prediction_probs = np.ndarray((size, 4))
        for i in range(size):
            temp_prediction_probs[i] = self.forests[1].predict(point=prediction_probs[i], soft=True).values()

        # Train third RF hierarchy
        self.forests[2].grow(points=temp_prediction_probs, responses=training_labels[:, 2])

    def test(self, test_labels, test_examples, num_classes, top_n=[1, 5]):
        if num_classes < np.max(top_n):
            raise AttributeError, "The number of classes must be greater than or equal to all of the top-n values during testing."

        size = len(test_labels)

        top_n_hits = {}
        for t_n in top_n:
            top_n_hits[t_n] = np.ndarray(size)

        for i in range(size):
            prediction_probs = self.forests[2].predict(point=test_examples[i], soft=True)
            sorted_class_probs = sorted(prediction_probs.items(), key=operator.itemgetter(1), reverse=True)

            for t_n in top_n_hits.keys():
                top_n = [c[0] for c in sorted_class_probs[0:t_n]]

                # Record prediction accuracy
                if test_labels[i, 2] in top_n:
                    top_n_hits[t_n][i] = True
                else:
                    top_n_hits[t_n][i] = False

        # Produce statistics dictionary
        stats = {}
        for key, val in top_n_hits.iteritems():
            stats['top %d accuracy' % key] = len([1 for p in val if p == True]) / float(size)

        return stats

    def optimize(self, training_examples, training_labels):
        raise NotImplementedError, "The HierarchicalForest.optimize() function has yet to be implemented."
        
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