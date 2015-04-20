import os
import sys
import struct
from array import array
import numpy as np


def read_mnist(path, dataset):
    # Determine whether to pull training or testing data
    if dataset == 'training':
        filename_img = os.path.join(path, 'train-images.idx3-ubyte')
        filename_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == 'testing':
        filename_img = os.path.join(path, 't10k-images.idx3-ubyte')
        filename_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "Dataset must be 'testing' or 'training'"

    # Read and save labels
    with open(filename_lbl, 'rb') as f:
        magic_num, size = struct.unpack(">II", f.read(8))
        label_raw = array('b', f.read())

    # Read and save images
    with open(filename_img, 'rb') as f:
        magic_num, size, rows, cols = struct.unpack(">IIII", f.read(16))
        image_raw = array('B', f.read())

    N = size
    labels = np.zeros(shape=N, dtype='int8')
    images = np.zeros(shape=(N, rows*cols), dtype='uint8')
    for i in range(N):
        labels[i] = label_raw[i]
        images[i] = np.array(image_raw[i*rows*cols : (i + 1)*rows*cols])

    return labels, images

if __name__ == "__main__":
    if len(sys.argv) !=3:
        print "Please provide valid input parameters ([path to MNIST files] ['training' or 'testing'])"
        exit(1)
    path = sys.argv[1]
    dataset = sys.argv[2]

    # Convert MNIST binary to image and save to file
    labels, images = read_mnist(path=path, dataset=dataset)
    print labels
    print images
