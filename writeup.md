# **Traffic Sign Recognition** 

## Project Writeup

[//]: # (Image References)

[image1]: ./figures/nominal_dataset.jpg "Nominal Dataset"
[image2]: ./figures/nominal_dataset_histogram.jpg "Nominal Dataset Histogram"
[image3]: ./figures/augmented_dataset.jpg "Augmented Dataset"
[image4]: ./figures/augmented_dataset_histogram.jpg "Augmented Dataset Histogram"

---
## Data Analysis

#### Basic summary, exploratory visualization and analysis of the of the initial data set. 

The original dataset for this project is the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The dataset provides three pickle files, each of them containing a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.

The three datasets are separated for Training, Validation and Test.
Using some simple numpy libraries, the analysis of the data showed:

 
| Original Dataset Feature         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
|Number of training examples    |34799|
|Number of validation examples   |4410|
|Number of testing examples   |12630|
|Image data shape   |(32, 32, 3)||
|Number of classes   |43|



Afterwards, I looked at an example of each picture (taking the first occurrence of each one of the 43 classess) and their distribution:

![alt text][image1]

![alt text][image2]


The fundamental findings for me, after this, were:

1. The dataset is significantly unbalanced, with some signs more represented than the others (there are 2010 instances of the 'Speed Limit (50 km/h)' sign and only 180 for the 'Speed Limit (20 km/h)' one.
2. Some of the images seem to be very dark.

This led me to try to artificially "augment" the dataset. 

#### Data set augmentation.

Two types of modification were applied to the original images in the dataset:

1. A random rotation in the range \[-20;20 \] deg
2. A random increment in intensity of the image, multiplying some of them for a facto up  to 1.8

In terms of size, my target was to create a dataset evenly distributed, with every sign as represented as the oringinal most common one, so for every class I have generated a number of image equal to the difference between their original occurrence and 2010 (again, the num. of instances of the most common sign).

The code I used is attached as part of the python notebook; the augmented dataset in output was saved as an additional pikle file with a dictionary containing only the  `'features'` and  `'labels'` key-value pairs.
The results in terms of dataset size and visual representation are:


| Augmented Dataset Feature         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
|Number of training examples    |86388|


![alt text][image3]

![alt text][image4]

---

## Network Architecture and Design

#### Data Preprocessing. 

After experimenting with running against grayscale images, I could not really appreciate a difference in the outcome of the model, so I finally resorted to stick with the RGB format. This made the pre-processing actually fairly simple, consisting in just a normalization of the images to transform them in the \[-128,128\] range.

#### Network Architecture. 

The starting point for this project was the LeNet architecture and its implementation provided as part of the Udacity [lab](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb).

It was interesting to me to verify how without changing the fundamental architecture, but manipulating the hyperparameters (number of features for each layer, number of epochs...) was it possible to already reach an accuracy of more than 91%.

In order to reach higher values, but also to experiment with different architectures, I started to modify the original design and try different approaches. While doing research on the subject, a good reference I found was [this](https://github.com/ericlavigne/CarND-Traffic-Sign-Classifier) project of a former udacity student, that pushed me towards the idea of increasing the number of "full" layers.
I tried different solution in this sense, and one of the things that I found was the tendedncy of the model towards some degree of overfitting, reaching accuracies of over 97% in training, but remaining consistently lower in testing. In order to limit the behavior, I have implemented several layers of dropout.

My final design can be described as it follows:


| Layer         		|     Description	        					|      Output|
|:---------------------:|:---------------------------------------------:|:---------------------:|
|Input    | Original Image | 32x32x3 RGB Image |
|Pre-processing    | Normalize Data | 32x32x3 Normalized RGB Image |
|Convolution    | 5x5, Depth 6  | 28x28x6 |
|Convolution    | 5x5, Depth 12  | 24x24x12 |
|Dropout    | 50 %  | 24x24x12 |
|Convolution    | 5x5, Depth 20  | 20x20x20 |
|Convolution    | 5x5, Depth 30  | 16x16x30 |
|Dropout    | 50 %  | 16x16x30 |
|Convolution    | 5x5, Depth 45  | 12x12x45|
|Convolution    | 5x5, Depth 70  | 8x8x70 |
|Dropout    | 50 %  | 8x8x70 |
|Flatten    | Transition from convolutional to dense  | 4480 |
|Dense    | Depth 400  | 400 |
|Dense    | Depth 100  | 100 |
|Dropout    | 50 %  | 100 |
|Dense    | Depth 43  | 43 (Final classifier) |

Furthermore:

* All the activation functions are RELUs.
* I used an Adam optimizer, with a contnt learning rate of 0.001. I have experimenting trying to manipulate this, and even implementing an exponential decay (some of the [references](https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf) I found while researching seemed to suggest an influence of that on the Adam algorythm also). I could actually find some benefit in that in some of the architecture I have tried, but ultimately it didn't seem to affect the final design, so I left it constant.

Given its "density", the model was trained on an AWS GPU instance, leading to a final accuracy of, in a time 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


