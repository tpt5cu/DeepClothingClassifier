#Script to predict image

import os
import numpy as np 
import glob 
from os.path import join as pJoin
import random
from keras.applications import vgg19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import ssl
import h5py

#Create new model
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model

def predictionCategoryLookup(pred):
	pred = int(pred)
	return categories[pred+1].rstrip()

def import_image(imgpath):
	# load an image in PIL format
	original = load_img(imgpath, target_size=(224, 224))
	# convert the PIL image to a numpy array
	# IN PIL - image is in (width, height, channel)
	# In Numpy - image is in (height, width, channel)
	numpy_image = img_to_array(original)
	return numpy_image

def predictClasses(img_Path):
	#IN DEV: Find probability of article of clothing
	# for file in test_labels[0]:
	img = import_image(img_Path)
	img = np.expand_dims(img, axis=0)
	pred = model.predict_proba(img)

	probs = []
	for i in pred[0]:
	    probs.append(i)
	probs = np.array(probs)
	bestChoice = np.argmax(probs)
	probsIndices = probs.argsort()
	top3indices=probsIndices[-8:]
	img = import_image(img_Path)
	plt.imshow(np.uint8(img))
	plt.show()
	#Print class number, article of clthing + category type, and the probability
	for i in top3indices:
	    print probs[i], predictionCategoryLookup(i)
	    
	return bestChoice, predictionCategoryLookup(bestChoice), max(probs)



file = ('categories.txt')
f=open(file, "r")
categories=f.readlines()


# Create the model
model = models.Sequential()

vgg_model = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Add the vgg convolutional base model
model.add(vgg_model)

model = load_model('Clothing_Classifier_Model.h5')


response = predictClasses('images/elliot.jpg')

print response




