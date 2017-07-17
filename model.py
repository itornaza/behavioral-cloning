#-------------
# Imports
#-------------

import os
import csv
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Input, Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#-------------
# Constants
#-------------

csv_path = './data_provided/driving_log.csv'
imag_path = './data_provided/IMG/'
recovery_path = './data_recovery/driving_log.csv'
recovery_imag_path = './data_recovery/IMG/'

class CSV_Headers:
    Center, Left, Right, Steering, Throttle, Brake, Speed = range(7)

#-------------
# Globals
#-------------

# Dataset lists
images = []
angles = []

# Debug controls
DEBUG = True
STATISTICS = False
FLIP_IMAGES = True
LEFT_IMAGES = True
RIGHT_IMAGES = True

#-----------------
# Hyperparameters
#-----------------

BATCH_SIZE = 128

TEST_SIZE = 0.2
EPOCHS = 3
LOSS = 'mse'
OPTIMIZER = 'adam'
KEEP_PROB = 0.5
ACTIVATION = 'relu'

# 0.2 was bad
# 0.35 was bad
# 0.33 was good
# 0.31 was awesome
STEERING_CORRECTION = 0.31
CUT_OFF_ANGLE = 0.1

#-------------
# Functions
#-------------

def appendData(image, angle):
    """
    Add the image and coresponding steering angle to the dataset
    """
    global images
    global angles
    images.append(image)
    angles.append(angle)

def flipImage(image, angle):
    """
    Given an image and it's steering angle return the flipped image and the corresponding steering angle
    """
    flipped_image = cv2.flip(image, 1)
    flipped_angle = angle * -1.0
    return (flipped_image, flipped_angle)

def showDebugInfo():
    """
    Report debug parameters
    """
    if DEBUG and FLIP_IMAGES: print("> Flipping images - enabled")
    if DEBUG and not FLIP_IMAGES: print("> Flipping images - disabled")
    if DEBUG and LEFT_IMAGES: print("> Processing left images - enabled")
    if DEBUG and not FLIP_IMAGES: print("> Flipping images - disabled")
    if DEBUG and RIGHT_IMAGES: print("> Processing right images - enabled")
    if DEBUG and not FLIP_IMAGES: print("> Flipping images - disabled")
    if STATISTICS: print("> Statistics - enabled")
    if not STATISTICS: print("> Statistics - disabled")

def prepareDataFromFile():
    """
    Parse the driving_log.csv file and return an list with the images and angle values.
    Dataset augmentation is taking place as well for flipped, left and right images
    """
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
                appendData(center_image, center_angle)

            # Add the flipped image as well to the dataset (only for the center images)
            if FLIP_IMAGES:
                (flipped_image, flipped_angle) = flipImage(center_image, center_angle)
                if abs(flipped_angle) > CUT_OFF_ANGLE:
                    if flip_flag:
                        # Keep only the 50% of the flipped images controlled by the flip_flag
                        appendData(flipped_image, flipped_angle)
                        flip_flag = False
                    else:
                        flip_flag = True

            # Add the left image with correction in angle to the dataset
            if LEFT_IMAGES:
                left_image_name = imag_path + line[CSV_Headers.Left].split('/')[-1]
                left_image = cv2.imread(left_image_name)
                left_angle = float(line[CSV_Headers.Steering]) + STEERING_CORRECTION
                appendData(left_image, left_angle)

            # Add right image with correction in angle to the dataset
            if RIGHT_IMAGES:
                right_image_name = imag_path + line[CSV_Headers.Right].split('/')[-1]
                right_image = cv2.imread(right_image_name)
                right_angle = float(line[CSV_Headers.Steering]) - STEERING_CORRECTION
                appendData(right_image, right_angle)

        # Report data size in debug mode
        if DEBUG: print("Number of images: " + str(len(images)) + " and number of angles: " + str(len(angles)))

def prepareRecoveryData():
    with open(recovery_path) as csvfile:
        reader = csv.reader(csvfile)
        
        # Control to get only the 50% of the flipped images
        flip_flag = True
        for line in reader:
            center_image_name = recovery_imag_path + line[CSV_Headers.Center].split('/')[-1]
            
            # Add the center image to the dataset
            center_image = cv2.imread(center_image_name)
            center_angle = float(line[CSV_Headers.Steering])
            if abs(center_angle) > CUT_OFF_ANGLE:
                appendData(center_image, center_angle)
        
        # Report data size in debug mode
        if DEBUG:
            print("Number of images plus recovery: " + str(len(images)) +
                  " and number of angles plus recovery: " + str(len(angles)))

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

# Deprecated
def createSimpleModel():
    """
    Just used to test that the pipeline works
    """
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

# Working model
def createModel():
    """
    Creates a model based comprized of 3 convolution and 2 fully connected layers
    """
    model = Sequential()
    
    # Preprocessing layers
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    
    # Convolution layers
    model.add(Convolution2D(32, 3, 3, activation = ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32, 3, 3, activation = ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, 3, 3, activation = ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation(ACTIVATION))
    model.add(Dropout(KEEP_PROB))
    model.add(Dense(1))
    
    # Compile and return
    model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics = [LOSS])
    return model

def showStatistics(training_history):
    """
    Display the statistice for the model trainning
    """
    # Print the keys contained in the history object
    print(training_history.history.keys())
    
    # Plot the training and validation loss for each epoch
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def trainModel(model, train_generator, train_samples, valid_generator, valid_samples):
    """
    Trains and saves the model as model.h5 to be used with the simulator
    """
    start = time.time()
    training_history = model.fit_generator(train_generator,
                                         samples_per_epoch = train_samples,
                                         validation_data = valid_generator,
                                         nb_val_samples = valid_samples,
                                         nb_epoch = EPOCHS,
                                         verbose = 1)
    end = time.time()

    # Show timing and model statistics
    if STATISTICS:
        print("Trainined " + str(EPOCHS) + " epochs in " + str(end - start) + " seconds")
        showStatistics(training_history)
    
    # Save the model
    model.save('model.h5')

#-------------
# Main
#-------------
if __name__ == '__main__':
    showDebugInfo()
    prepareDataFromFile()
    prepareRecoveryData()
    x_train, x_valid, y_train, y_valid = train_test_split(images, angles, test_size = TEST_SIZE)
    train_generator = generator(x_train, y_train, batch_size = BATCH_SIZE)
    valid_generator = generator(x_valid, y_valid, batch_size = BATCH_SIZE)
    model = createModel()
    trainModel(model, train_generator, len(y_train), valid_generator, len(y_valid))
