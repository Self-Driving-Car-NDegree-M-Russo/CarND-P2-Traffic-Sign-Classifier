# **Traffic Sign Recognition** 

## Project Writeup

[//]: # (Image References)

[image1]: ./figures/nominal_dataset.jpg "Nominal Dataset"
[image2]: ./figures/nominal_dataset_histogram.jpg "Nominal Dataset Histogram"
[image3]: ./figures/augmented_dataset.jpg "Augmented Dataset"
[image4]: ./figures/augmented_dataset_histogram.jpg "Augmented Dataset Histogram"
[image5]: ./figures/web_dataset.jpg "Images from the Web"

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
* I used an Adam optimizer, with a constant learning rate of 0.001. I have experimenting trying to manipulate this, and even implementing an exponential decay (some of the [references](https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf) I found while researching seemed to suggest an influence of that on the Adam algorythm also). I could actually find some benefit in that in some of the architecture I have tried, but ultimately it didn't seem to affect the final design, so I left it constant.
* After trying various combinations, i decided to settle for 20 Epochs. Here too, the compromise was mostly between performances against the test data and overfitting of the training data.

Given its "density", the model was trained on an AWS GPU instance, leading to a final accuracy of 93.4%, in a time of almost 12 minutes.

While satusfactory from a project's perspective, I think it's fair to say that the accuracy shown against the training dataset is still fairly better (surpassing 96%), whcih means that probably steps can still be taken to improve the resiliency of the design.

---

## Evaluation of the model against new images

6 new images of German traffic signs have been collected from the internet, with different degrees of resolution and clarity:

![alt text][image5]


These images have then be gven in input to the model, given as a result a global accuracy score of 83% (5 images have been identified out of 6).

For the _non_ identified images, the top 5 softmax probablities identified by the model were:

| Signal         		|     Probablity	        					| 
|:---------------------:|:---------------------------------------------:| 
|Speed limit (30 Km/h)   | *Real Signal* |
|Right-of-way at the next intersection   | 0.96 |
|Vehicles over 3.5 metric tons prohibited   | 0.019 |
|Speed limit (50km/h)  | 0.015 |
|Speed limit (30km/h)   | 0.002 |
|No entry   | 0.001 |

| Signal         		|     Probablity	        					| 
|:---------------------:|:---------------------------------------------:| 
|Speed limit (30 Km/h)   | *Real Signal* |
|Right-of-way at the next intersection   | 0.96 |
|Vehicles over 3.5 metric tons prohibited   | 0.019 |
|Speed limit (50km/h)  | 0.015 |
|Speed limit (30km/h)   | 0.002 |
|No entry   | 0.001 |


---


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


