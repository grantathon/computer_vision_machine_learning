import os
import sys
import time
from HierarchicalRandomForest_keraudren import *
import numpy as np
import json
import multiprocessing as mp
from pprint import pprint

if len(sys.argv) != 5:
    print "Please provide valid input parameters ([path to files] [training count] [testing count] [optimize])\n"
    exit(1)

script_start_time = time.time()

# User inputs
path = sys.argv[1]
training_cnt = int(sys.argv[2])
testing_cnt = int(sys.argv[3])
optimize = bool(int(sys.argv[4]))

# Constants
NUM_PROCS = mp.cpu_count()
NUM_FEATURES = 4096
HIGH_LEVEL_CLASS_TO_INDEX_FILE = "high_level_class_to_index_map.json"
LOW_LEVEL_CLASS_TO_INDEX_FILE = "low_level_class_to_index_map.json"
CATEGORY_TO_INDEX_FILE = "category_to_index_map.json"
PLACES_CAT_LABELS_FILE = "places_category_labels.csv"

print "Reading the configuration files..."
start_time = time.time()

# Initialize high-level class to index mappings
with open(HIGH_LEVEL_CLASS_TO_INDEX_FILE) as f:
    high_level_idx_map = json.loads(f.read())

# Initialize low-level class to index mappings
with open(LOW_LEVEL_CLASS_TO_INDEX_FILE) as f:
    low_level_idx_map = json.loads(f.read())

# Initialize category to index mappings
with open(CATEGORY_TO_INDEX_FILE) as f:
    category_idx_map = json.loads(f.read())

# Import csv with label information and build label map
label_data = np.genfromtxt(PLACES_CAT_LABELS_FILE, delimiter=',', dtype=str)
label_map = {}
for i in range(0, len(label_data)):
    label_map[label_data[i, 0]] = label_data[i, 1:]

end_time = time.time()
print "Finished reading the configuration files in %f seconds.\n" % (end_time - start_time)

# Load training and testing data
print 'Reading the training and testing data...'
start_time = time.time()

training_offset = 0
testing_offset = 0
for root, dir, files in os.walk(path):
    num_files = len(files)

    # Skip directories without files
    if num_files == 0:
        continue

    train_examples = np.ndarray((num_files*training_cnt, NUM_FEATURES))
    train_labels = np.ndarray((num_files*training_cnt, 3))
    test_examples = np.ndarray((num_files*testing_cnt, NUM_FEATURES))
    test_labels = np.ndarray((num_files*testing_cnt, 3))

    for data_file in files:
        file_path = "./" + path + data_file
        
        # Read in training and testing examples
        data = np.load(file_path)
        train_examples[training_offset:(training_offset+training_cnt), :] = data[0:training_cnt, :]
        test_examples[testing_offset:(testing_offset+testing_cnt), :] = data[training_cnt:(training_cnt+testing_cnt), :]
        
        # Assign labels for training and testing examples
        filename_wo_ext = os.path.splitext(data_file)[0]
        train_labels[training_offset:(training_offset+training_cnt)] = [[high_level_idx_map[label_map[filename_wo_ext][0]], low_level_idx_map[label_map[filename_wo_ext][1]], category_idx_map[filename_wo_ext]] for i in range(0, training_cnt)]
        test_labels[testing_offset:(testing_offset+testing_cnt)] = [[high_level_idx_map[label_map[filename_wo_ext][0]], low_level_idx_map[label_map[filename_wo_ext][1]], category_idx_map[filename_wo_ext]] for i in range(0, testing_cnt)]

        training_offset += training_cnt
        testing_offset += testing_cnt

# Replace all NaNs and infs with zeros
train_examples[np.isnan(train_examples)] = 0
train_examples[np.isinf(train_examples)] = 0
test_examples[np.isnan(test_examples)] = 0
test_examples[np.isinf(test_examples)] = 0

end_time = time.time()
print "Finished reading the training and testing data in %f seconds.\n" % (end_time - start_time)

# Build, train, and test the hierarchical RF classifier
classifier = HierarchicalRandomForest(n_estimators=100, n_procs=NUM_PROCS)

print 'Training the hierarchical random forest classifier...'
start_time = time.time()
classifier.train(training_labels=train_labels, training_examples=train_examples, num_classes=num_files, optimize=optimize)
end_time = time.time()
print "Finished training the hierarchical random forest classifier in %f seconds.\n" % (end_time - start_time)

print 'Testing the hierarchical random forest classifier...'
start_time = time.time()
pred_results = classifier.test(test_labels=test_labels, test_examples=test_examples, num_classes=num_files)
end_time = time.time()
print "Finished testing the hierarchical random forest classifier in %f seconds.\n" % (end_time - start_time)

# Output results
print "Accuracy results:"
for key, val in pred_results.iteritems():
    print "  %s: %.4f" % (key, val)
print

script_end_time = time.time()
print "Finished running script in %f seconds." % (script_end_time - script_start_time)