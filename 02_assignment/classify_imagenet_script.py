import os
from os.path import expanduser
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print 'Please provide input arguements {[image URI]}'
		exit(1)

	img_uri = sys.argv[1]

	# Make sure that caffe is on the python path:
	caffe_root = expanduser("~") + '/caffe/'  # this file is expected to be in {caffe_root}/examples
	sys.path.insert(0, caffe_root + 'python')

	import caffe

	# Set the right path to your model definition file, pretrained model weights,
	# and the image you would like to classify.
	MODEL_FILE = '/usr/prakt/p045/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
	PRETRAINED = '/usr/prakt/p045/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

	# import os
	if not os.path.isfile(PRETRAINED):
	    print("Downloading pre-trained CaffeNet model...")
	    sys.argv.append(caffe_root + 'models/bvlc_reference_caffenet')
	    execfile(caffe_root + 'scripts/download_model_binary.py')

	# Setup neural network classifier
	# caffe.set_mode_cpu()
	caffe.set_mode_gpu()
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
	                       channel_swap=(2,1,0),
	                       raw_scale=255,
	                       image_dims=(256, 256))

	# Display input image
	input_image = caffe.io.load_image(img_uri)
	plt.subplot(2, 1, 1)
	plt.imshow(input_image)
	plt.title('Input Image and Class Prediction Histogram')

	# Predict the classification of the image and display stats
	prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
	print 'predicted class:', prediction[0].argmax()
	
	# Display class prediction histogram
	plt.subplot(2, 1, 2)
	plt.plot(prediction[0])
	plt.ylabel('Probability')
	plt.xlabel('Class Index')

	plt.show()
