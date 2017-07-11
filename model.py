#-------------
# Imports
#-------------

import os
import csv
import cv2
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#-------------
# Constants
#-------------

csv_path = './data_provided/driving_log.csv'
imag_path = './data_provided/IMG/'

class CSV_Headers:
    Center, Left, Right, Steering, Throttle, Brake, Speed = range(7)

#-------------
# Parameters
#-------------

BATCH_SIZE = 128
TEST_SIZE = 0.2
EPOCHS = 1
LOSS = 'mse'
OPTIMIZER = 'adam'
KEEP_PROB = 0.5
ACTIVATION = 'elu'
STEERING_CORRECTION = 0.2
CUT_OFF_ANGLE = 0.1

DEBUG = True
FLIP_IMAGES = False
LEFT_IMAGES = False
RIGHT_IMAGES = False

#-------------
# Functions
#-------------

def getDataFromFile():
    """
    Parse the driving_log.csv file and return an list with the images and angle values.
    Dataset augmentation is taking place as well for flipped, left and right images
    """
    
    if DEBUG and FLIP_IMAGES: print("Flipping images ...")
    if DEBUG and LEFT_IMAGES: print("Processing left images ...")
    if DEBUG and RIGHT_IMAGES: print("Processing right images ...")
    
    images, angles = [], []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
            
        # Control to get only the 50% of the flipped images
        flip_flag = True
        for line in reader:
            center_image_name = imag_path + line[CSV_Headers.Center].split('/')[-1]
        
            # Add the center image to the dataset
            center_image = cv2.imread(center_image_name)
            center_angle = float(line[CSV_Headers.Steering])
            
            if abs(center_angle) > CUT_OFF_ANGLE:
                images.append(center_image)
                angles.append(center_angle)

            # Add the flipped image as well to the dataset (only for the center images)
            if FLIP_IMAGES:
                flipped_image = cv2.flip(center_image, 1)
                flipped_angle = center_angle * -1.0
                
                if abs(flipped_angle) > CUT_OFF_ANGLE:
                    if flip_flag:
                        images.append(flipped_image)
                        angles.append(flipped_angle)
                        flip_flag = False
                    else:
                        flip_flag = True

            # Add the left image with correction in angle to the dataset
            if LEFT_IMAGES:
                left_image_name = imag_path + line[CSV_Headers.Left].split('/')[-1]
                left_image = cv2.imread(left_image_name)
                left_angle = float(line[CSV_Headers.Steering]) + STEERING_CORRECTION
                images.append(left_image)
                angles.append(left_angle)

            # Add right image with correction in angle to the dataset
            if RIGHT_IMAGES:
                right_image_name = imag_path + line[CSV_Headers.Right].split('/')[-1]
                right_image = cv2.imread(right_image_name)
                right_angle = float(line[CSV_Headers.Steering]) - STEERING_CORRECTION
                images.append(right_image)
                angles.append(right_angle)

    return (images, angles)

def generator(images, angles, batch_size = BATCH_SIZE):
    """
    Coprocess to provide images and angle data in batches
    """
    num_samples = len(images)
    
    # Get a random set of indices with the length of the batch size
    random_index = np.random.choice(num_samples, size = BATCH_SIZE, replace = False)

    # Forever return the batch of images and angles
    while True:
        images_batch = []
        angles_batch = []
        for index in random_index:
            images_batch.append(images[index])
            angles_batch.append(angles[index])
        
        # Make the images and angles numpy arrays ready for Keras
        X_train = np.array(images_batch)
        y_train = np.array(angles_batch)

        # Return batch
        yield shuffle(X_train, y_train)

# Deprecated - Just used to test that the pipeline works
def createSimpleModel():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

# Working model
def createIonModel():
    """
    Creates a Keras model based on the NVIDIA architecture with Dropout layers to reduce overfitting.
    The NVIDIA fully connected 1164 layer has been removed due to AWS excessive memory issues.
    """
    model = Sequential()
    
    # Preprocessing
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    
    # Convolution layers
    model.add(Convolution2D(24, 5, 5, activation = ACTIVATION))
    model.add(Convolution2D(36, 5, 5, activation = ACTIVATION))
    model.add(Convolution2D(48, 5, 5, activation = ACTIVATION))
    model.add(Convolution2D(64, 3, 3, activation = ACTIVATION))
    model.add(Convolution2D(64, 3, 3, activation = ACTIVATION))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation = ACTIVATION))
    model.add(Dropout(KEEP_PROB))
    model.add(Dense(50, activation = ACTIVATION))
    model.add(Dropout(KEEP_PROB))
    model.add(Dense(10, activation = ACTIVATION))
    model.add(Dropout(KEEP_PROB))
    model.add(Dense(1))
    
    return model

def trainModel(model, train_generator, train_samples, valid_generator, valid_samples):
    """
    Trains and saves the model as model.h5 to be used with the simulator
    """
    
    # Time the training
    start = time.time()

    model.compile(loss = LOSS, optimizer = OPTIMIZER)
    model.fit_generator(train_generator,
                        samples_per_epoch = train_samples,
                        validation_data = valid_generator,
                        nb_val_samples = valid_samples,
                        nb_epoch = EPOCHS)
    end = time.time()
    print("Trainined " + str(EPOCHS) + " epochs in " + str(end - start) + " seconds")

    model.save('model.h5')

#-------------
# Main
#-------------
if __name__ == '__main__':
    (images, angles) = getDataFromFile()
    if DEBUG: print("Number of images: " + str(len(images)) + " and number of angles: " + str(len(angles)))
    x_train, x_valid, y_train, y_valid = train_test_split(images, angles, test_size = TEST_SIZE)
    train_generator = generator(x_train, y_train, batch_size = BATCH_SIZE)
    valid_generator = generator(x_valid, y_valid, batch_size = BATCH_SIZE)
    model = createIonModel()
    trainModel(model, train_generator, len(y_train), valid_generator, len(y_valid))
