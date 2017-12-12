# Behavioral Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image2]: ./examples/center_driving.jpg "Center line driving"
[image3]: ./examples/recover_1.jpg "Recovery Image"
[image4]: ./examples/recover_2.jpg "Recovery Image"
[image5]: ./examples/recover_3.jpg "Recovery Image"
[image6]: ./examples/flipped.jpg "Flipped Image"
[image7]: ./examples/loss.png "Mean Square Loss"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
`python drive.py model.h5`

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 64 (function createModel in the model.py file.) 

The model includes ReLU layers to introduce nonlinearity after all the convolution layers, and the data is normalized in the model using a Keras lambda layer (at the beginning of the createModel function). Max pooling layers were also implemented after each ReLU layer.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer with a default keep probability of 50% in order to reduce overfitting just after the fully connected layer. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data that udacity provided for center lane driving (found in the data_provided directory), and a set of data that I have capture for recovering from the left and right sides of the road (in the data_recovery directory).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a simple cnn compined with a set of fully connected layers with the last of them having an output of 1 to satisfy our regression problem. 

My first step was to use a convolution neural network model similar to the NVIDIA architecture I thought this model might be appropriate because it is already used in self driving cars. However, due to memory limitations I had to strip the model a bit in order to work on both Amazon Web Services EC2 GPU enabled servers as well as my laptop.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used 3 epochs to train the network as I have noticed that above epoch 4 overfitting occured. 

I have decided not to take into consideration any image that had a steering angle less than 0.1 that simulates centerline driving with minimal correction to steering. This led to having more evenly distributed dataset.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I have used the left and right camera images to simulate driving with offset from the center line. Furthermore, I augmented the dataset by capturing more data for recoverying the car from the boundaries. I have also flipped the center images and inversed the steering angle in order to have more data.

Another parameter that I had to look was the cut off angle that the car should use for recovery when the images from the left and right car cameras were used. I tested a lot of values with vary different results ranfging from 0.2 to 0.35. As a result I ended up using 0.31 for optimum results.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (createModel function in model.py) consisted of a convolution neural network with the following layers and layer sizes

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RBG images                          |
| Lamda                 | Normalize images to [-0.5, 0.5]               |
| Cropping 2D           | Remove 70 pixels from top and 20 from bottom  |
| Convolution2D         | 32 filter, 3x3 stride, valid padding          |
| ReLU					|                                               |
| Max pooling 2D      	| 2x2 stride, valid padding                     |
| Convolution2D         | 32 filter, 3x3 stride, valid padding          |
| ReLU					|                                               |
| Max pooling 2D      	| 2x2 stride, valid padding                     |
| Convolution2D         | 64 filter, 3x3 stride, valid padding          |
| ReLU					|                                               |
| Max pooling 2D      	| 2x2 stride, valid padding                     |
| Flatten               |                                               |	
| Fully connected		| Outputs 64                                    |
| ReLU                  |                                               |
| Dropout               | Keep prob 0.5                                 |
| Fully connected		| Outputs 1     								|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I relied on the default udacity data for center line driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to lean towords the center of the road. These images show what a recovery looks like starting from right towards the center line:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image6]

After the collection process, I had 20329 number of data points. This includes the 50% of the flipped center line images, and the recovery images. In addition, the images with steering less than 0.1 are not included. I then preprocessed this data by normalizing the ppixel values from [0 - 255] to [-0.5 to 0.5] in order to have zero mean and zero variance. Lastly, I had cropped the 70 top pixels and the bottom 20 pixels to ignore the image details above the horizon and the hood of the car.

I finally randomly shuffled the data set and put 80% of the data into a validation set containing 20% of the data. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the mean squared error in the validation set dropping to from 0.0369 in epoch 1 to 0.0364 in epoch 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The training set and validation set losses are shown in the following figure

![alt text][image7]

### Final video

The video of the self driving car performing a lap on the simulator can be found on youtube [driving video](https://youtu.be/bXWiGrzufB8)
