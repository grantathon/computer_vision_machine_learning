import os
import sys
import json
import numpy as np
from PIL import Image
from pprint import pprint

if len(sys.argv) != 4:
    print "Please provide required input parameters ([Places base directory] [output directory] [labels file])"
    exit(1)

# Global constants
IMG_DATA_SIZE = 131072
HIGH_LEVEL_CLASS_TO_INDEX_FILE = "high_level_class_to_index_map.json"
LOW_LEVEL_CLASS_TO_INDEX_FILE = "low_level_class_to_index_map.json"
CATEGORY_TO_INDEX_FILE = "category_to_index_map.json"

# Input parameters
places_dir = sys.argv[1]
output_dir = sys.argv[2]
labels_file = sys.argv[3]

# Initialize high-level class to index mappings
with open(HIGH_LEVEL_CLASS_TO_INDEX_FILE) as f:
    high_level_idx_map = json.loads(f.read())

# Initialize low-level class to index mappings
with open(LOW_LEVEL_CLASS_TO_INDEX_FILE) as f:
    low_level_idx_map = json.loads(f.read())

# Initialize category to index mappings
with open(CATEGORY_TO_INDEX_FILE) as f:
    category_idx_map = json.loads(f.read())

# Import csv with label information
label_data = np.genfromtxt(labels_file, delimiter=',', dtype=str)

# Process and store all label information as new data sets
for i in range(1, len(label_data)):
    cat_name = label_data[i, 0]
    high_level_name = label_data[i, 1]
    low_level_name = label_data[i, 2]
    cat_dir = "%s/%s/%s" % (places_dir, cat_name[0], cat_name)

    # Create binary file for coming examples
    output_file_name = "%s/%s.bin" % (output_dir, cat_name)
    output_file = open(output_file_name, 'wb')
    output_file.close()

    # Traverse through category directory
    for root, dir, files in os.walk(cat_dir):
        num_files = len(files)

        # Skip directories without files
        if num_files == 0:
            continue

        print "%s, %s, %s" % (high_level_name, low_level_name, cat_name)
        output_file = open(output_file_name, 'ab')

        # Label all example images according to csv with label info
        example = np.ndarray(IMG_DATA_SIZE+3)
        for j in range(0, num_files):
            img_path = "%s/%s" % (cat_dir, files[j])

            # Retrieve images and convert to grayscale
            img = Image.open(img_path).convert('LA')
            img_raw_data = img.tostring()

            for k in range(0, IMG_DATA_SIZE):
                example[k] = ord(img_raw_data[k])

            # example[0:IMG_DATA_SIZE] = img_data
            example[IMG_DATA_SIZE] = high_level_idx_map[high_level_name]
            example[IMG_DATA_SIZE+1] = low_level_idx_map[low_level_name]
            example[IMG_DATA_SIZE+2] = category_idx_map[cat_name]

            # Add image to binary file
            example_byte_array = bytearray(example)
            output_file.write(example_byte_array)

        output_file.close()
