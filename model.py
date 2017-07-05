
# Imports
import os
import csv
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Flatten, Dense
from keras.models import Sequential, Model

#-------------
# Constants
#-------------

CSV_PATH = './data_provided/driving_log.csv'
IMG_PATH = './data_provided/IMG/'
BATCH_SIZE = 32
TEST_SIZE = 0.2
EPOCHS = 1
LOSS = 'mse'
OPTIMIZER = 'adam'

#-------------
# Functions
#-------------

def getSamplesFromFile():
    """
    Parse the driving_log.csv file and return an list with the controls values
    """
    samples = []
    with open(CSV_PATH) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            samples.append(sample)
    return samples

def generator(samples, batch_size = BATCH_SIZE):
    """
    Coprocess to provide images and control data in batches
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image_name = IMG_PATH + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(image_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            
            # Trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            # Return batch
            yield shuffle(X_train, y_train)

def createSimpleModel():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

def createModel():
    return 0

def trainModel(model, train_generator, train_samples, valid_generator, valid_samples):
    """
    Trains and saves the model as model.h5 to be used with the simulator
    """
    model.compile(loss = LOSS, optimizer = OPTIMIZER)
    model.fit_generator(train_generator,
                        samples_per_epoch = len(train_samples),
                        validation_data = valid_generator,
                        nb_val_samples = len(valid_samples),
                        nb_epoch = EPOCHS)
    model.save('model.h5')

#-------------
# Pipeline
#-------------

samples = getSamplesFromFile()
train_samples, valid_samples = train_test_split(samples, test_size = TEST_SIZE)
train_generator = generator(train_samples, batch_size = BATCH_SIZE)
valid_generator = generator(valid_samples, batch_size = BATCH_SIZE)
model = createSimpleModel()
trainModel(model, train_generator, train_samples, valid_generator, valid_samples)
