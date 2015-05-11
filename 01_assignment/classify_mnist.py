from sklearn import svm, tree, ensemble, grid_search, cross_validation
import numpy as np


def train_classifier(clf_name, labels, examples, num_samples=None, optimize=False, cross_validate=False):
    if optimize:
        print 'Training the classifier w/optimization...\n'
        clf = optimize_classifier(opt_method='grid', clf_name=clf_name)
    else:
        print 'Training the classifier w/o optimization...\n'
        if clf_name == 'svm':
            clf = svm.SVC(kernel='poly', degree=2)
        elif clf_name == 'tree':
            clf = tree.DecisionTreeClassifier()
        elif clf_name == 'forest':
            clf = ensemble.RandomForestClassifier(criterion='entropy', max_depth=10, max_features='auto',
                                                  n_estimators=40)
        elif clf_name == 'adaboost':
            clf = ensemble.AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100, learning_rate=0.2)
        elif clf_name == 'gradientboost':
            clf = ensemble.GradientBoostingClassifier(learning_rate=0.2, n_estimators=150)
        else:
            raise ValueError, 'ERROR: Unknown classifier name %s' % (clf_name)

    if num_samples is None:
        num_samples = len(labels)

    clf.fit(examples[0:num_samples], labels[0:num_samples])

    # Cross-validate classifier
    if cross_validate:
        print 'Running k-fold cross-validation...\n'
        avg, std = cross_validate_classifier(labels=labels[0:num_samples], examples=examples[0:num_samples], clf=clf)
    else:
        avg = None
        std = None

    return clf, avg, std


def test_classifier(labels, examples, clf):
    size = len(labels)
    correct_predictions = np.ndarray(shape=size)

    print 'Testing the classifier...\n'
    for i in range(size):
        result = clf.predict(examples[i])

        # Record prediction accuracy
        if result == labels[i]:
            correct_predictions[i] = True
        else:
            correct_predictions[i] = False

    # Produce statistics dictionary
    stats = dict()
    stats['Prediction Accuracy'] = len([1 for p in correct_predictions if p == True]) / float(size)

    return stats


def optimize_classifier(opt_method, clf_name):
    if opt_method == 'grid':
        if clf_name == 'svm':
            param_grid = [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.0, 0.001, 0.0001], 'kernel': ['rbf']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.0, 0.001, 0.0001], 'kernel': ['sigmoid']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.0, 0.001, 0.0001], 'degree': [2, 3], 'kernel': ['poly']},
            ]

            estimator = svm.SVC()
        elif clf_name == 'tree':
            param_grid = [
                {'criterion': ['gini'], 'max_depth': [None, 5, 6, 7, 8, 9, 10],
                 'max_features': [None, 'auto', 'sqrt', 'log2']},
                {'criterion': ['entropy'], 'max_depth': [None, 5, 6, 7, 8, 9, 10],
                 'max_features': [None, 'auto', 'sqrt', 'log2']}
            ]

            estimator = tree.DecisionTreeClassifier()
        elif clf_name == 'forest':
            param_grid = [
                {'criterion': ['gini'], 'max_depth': [None, 5, 6, 7, 8, 9, 10], 'n_estimators': [10, 20, 30, 40, 50],
                 'max_features': [None, 'auto', 'sqrt', 'log2']},
                {'criterion': ['entropy'], 'max_depth': [None, 5, 6, 7, 8, 9, 10], 'n_estimators': [10, 20, 30, 40, 50],
                 'max_features': [None, 'auto', 'sqrt', 'log2']}
            ]

            estimator = ensemble.RandomForestClassifier()
        elif clf_name == 'adaboost':
            param_grid = [
                {'algorithm': ['SAMME'], 'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 'learning_rate': [0.2, 0.4, 0.6, 0.8, 1]},
                {'algorithm': ['SAMME.R'], 'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 'learning_rate': [0.2, 0.4, 0.6, 0.8, 1]}
            ]

            estimator = ensemble.AdaBoostClassifier()
        elif clf_name == 'gradientboost':
            param_grid = [
                {'loss': ['deviance'], 'n_estimators': [50, 100, 150, 200],
                 'learning_rate': [0.2, 0.4, 0.6, 0.8, 1], 'max_depth': [2, 3, 4, 5, 6, 7]},
                # {'loss': ['exponential'], 'n_estimators': [50, 100, 150, 200],
                #  'learning_rate': [0.2, 0.4, 0.6, 0.8, 1], 'max_depth': [2, 3, 4, 5, 6, 7]},
                # {'loss': ['deviance'], 'n_estimators': [50, 100, 150, 200, 250, 300],
                #  'learning_rate': [0.2, 0.4, 0.6, 0.8, 1], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
                # {'loss': ['exponential'], 'n_estimators': [50, 100, 150, 200, 250, 300],
                #  'learning_rate': [0.2, 0.4, 0.6, 0.8, 1], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
            ]

            estimator = ensemble.GradientBoostingClassifier()
        else:
            raise ValueError, 'Unexpected classifier name %s' % (clf_name)

        clf = grid_search.GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy')
    # elif opt_method == 'random':
    #     pass
    else:
        print "Unexpected optimization method %s" % (opt_method)
        exit(1)

    return clf


def cross_validate_classifier(labels, examples, clf, k=5):
    scores = cross_validation.cross_val_score(clf, examples, labels, cv=k, scoring='accuracy')

    return scores.mean(), scores.std()*2
