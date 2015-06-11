import os
import sys
import numpy as np

if len(sys.argv) != 4:
    print "Please provide required input parameters ([Places base directory] [output directory] [labels file])"
    exit(1)

# Input parameters
places_dir = sys.argv[1]
output_dir = sys.argv[2]
labels_file = sys.argv[3]

# Import csv with label information
label_data = np.genfromtxt(labels_file, delimiter=',', dtype=str)

# Process and store all label information as new data sets
for i in range(0, len(label_data)):
    cat_name = label_data[i, 0]
    high_level_name = label_data[i, 1]
    low_level_name = label_data[i, 2]
    cat_dir = "%s/%s/%s" % (places_dir, cat_name[0], cat_name)

    # Traverse through category directory
    for root, dir, files in os.walk(cat_dir):
        num_files = len(files)

        # Skip directories without files
        if num_files == 0:
            continue

        print "%s, %s, %s" % (high_level_name, low_level_name, cat_name)

        # Label all example images according to csv with label info
        cat_examples = np.ndarray((num_files, 4), dtype=object)
        for j in range(0, num_files):
            cat_examples[j, 0] = files[j]
            cat_examples[j, 1] = high_level_name
            cat_examples[j, 2] = low_level_name
            cat_examples[j, 3] = cat_name

        # Save labelled examples to output directory
        output_file = "%s/%s.csv" % (output_dir, cat_name)
        np.savetxt(output_file, cat_examples, delimiter=',', fmt="%s")

