import os
from os.path import expanduser
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print 'Please provide input arguements {[image URI]}'
		exit(1)

	img_uri = sys.argv[1]

	# Make sure that caffe is on the python path:
	home_dir = expanduser("~")
	caffe_root = home_dir + '/caffe'  # this file is expected to be in {caffe_root}/examples
	sys.path.insert(0, caffe_root + '/python')

	import caffe

	# Set the right path to your model definition file, pretrained model weights,
	# and the image you would like to classify.
	MODEL_FILE = caffe_root + '/models/bvlc_reference_caffenet/deploy.prototxt'
	PRETRAINED = caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

	if not os.path.isfile(PRETRAINED):
	    print("Downloading pre-trained CaffeNet model...")
	    sys.argv.append(caffe_root + '/models/bvlc_reference_caffenet')
	    execfile(caffe_root + '/scripts/download_model_binary.py')

	# Setup neural network classifier
	# caffe.set_mode_cpu()
	caffe.set_mode_gpu()
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	                       mean=np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
	                       channel_swap=(2,1,0),
	                       raw_scale=511,
	                       image_dims=(256, 256))

	# Display input image
	input_image = caffe.io.load_image(img_uri)
	plt.subplot(2, 1, 1)
	plt.imshow(input_image)
	plt.title('Input Image and Class Prediction Histogram')

	# Predict the classification of the image
	prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically

	# Display class prediction histogram based on softmax probability
	plt.subplot(2, 1, 2)
	plt.plot(prediction[0])
	plt.ylabel('Probability')
	plt.xlabel('Class Index')
	
	# Retrieve the name and ID for each class
	with open(caffe_root + '/data/ilsvrc12/synset_words.txt') as f:
	    labels_df = pd.DataFrame([
	        {
	            'synset_id': l.strip().split(' ')[0],
	            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
	        }
	        for l in f.readlines()])

	# Join predictions with proper names and IDs
	labels_df.sort('synset_id')
	predictions_df = pd.DataFrame(data=prediction[0], columns=['softmax'])
	predictions_df = predictions_df.join(labels_df)

	# Find the top prediction and display stats
	prediction = predictions_df.iloc[prediction[0].argmax()]
	prediction_entropy = -np.array([e*np.log(e) for e in predictions_df['softmax']]).sum()
	print '\nPrediction:'
	print '    Class name:    %s' % (prediction['name'])
	print '    Synset ID:     %s' % (prediction['synset_id'])
	print '    Softmax:       %f' % (prediction['softmax'])
	print '    Entropy:       %f' % (prediction_entropy)

	plt.show()
