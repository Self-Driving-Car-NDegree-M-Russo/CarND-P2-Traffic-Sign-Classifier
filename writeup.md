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

The dataset is provided as three pickle files, each of them containing a dictionary with 4 key/value pairs:

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
2. A random increment in intensity of the image, multiplying some of them for a factor up  to 1.8

In terms of size, my target was to create a dataset evenly distributed, with every sign as represented as the oringinal most common one, so for every class I have generated a number of image equal to the difference between their original occurrence and 2010 (again, the num. of instances of the most common sign).

The code I used is attached as part of the python notebook; the augmented dataset in output was saved as an additional pickle file with a dictionary containing only the  `'features'` and  `'labels'` key-value pairs.
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

It was interesting for me to verify how without changing the fundamental architecture, but manipulating the hyperparameters (number of features for each layer, number of epochs...) it was possible to already reach an accuracy of more than 91%.

In order to reach higher values, but also to experiment with different architectures, I started to modify the original design and try different approaches. While doing research on the subject, a good reference I found was [this](https://github.com/ericlavigne/CarND-Traffic-Sign-Classifier) project of a former Udacity student, that pushed me towards the idea of increasing the number of "full" layers.
I tried different solutions in this sense, and one of the things that I found was the tendency of the model towards some degree of overfitting, reaching accuracies of over 97% in training, but remaining consistently lower in testing. In order to limit the behavior, I have implemented several layers of dropout.

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
* I used an Adam optimizer, with a constant learning rate of 0.001. I have experimented trying to manipulate this, and even implementing an exponential decay (some of the [references](https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf) I found while researching seemed to suggest an influence of that on the Adam algorythm also). I could actually find some benefit in that in some of the architectures I have tried, but ultimately it didn't seem to affect the final design, so I finally left it constant.
* After trying various combinations, I decided to settle for 20 Epochs. Here too, the compromise was mostly between performances against the test data and overfitting of the training data.

Given its "density", the model was trained on an AWS GPU instance, leading to a final accuracy of 93.9%, in a time of almost 12 minutes.

While satusfactory from a project's perspective, I think it's fair to say that the accuracy shown against the training dataset is still fairly better (surpassing 96% in few runs), which means that probably steps can still be taken to improve the resiliency of the design.

---

## Evaluation of the model against new images

6 new images of German traffic signs have been collected from the internet, with different degrees of resolution and clarity:

![alt text][image5]


These images have then be gven in input to the model, given as a result a global accuracy score of 66.7% (4 images have been identified out of 6).

For the _non_ identified images, the top 5 softmax probablities identified by the model were:

| Signal         		|     Probablity	        					| 
|:---------------------:|:---------------------------------------------:| 
|Speed limit (20 Km/h)   | *Real Signal* |
|General caution   | 0.719 |
|Traffic Signals | 0.19 |
|Pedestrians  | 0.002 |
|Dangerous curve to the right   | 0.002 |
|Dangerous curve to the left  | 0.001 |


| Signal         		|     Probablity	        					| 
|:---------------------:|:---------------------------------------------:| 
|No Entry   | *Real Signal* |
|Speed limit (20 Km/h)   | 0.99 |
|No Entry   | 6.8e-04 |
|Priority road  | 7.3e-05 |
|Road work   |  7.05e-05 |
|Dangerous curve to the right   | 4.48e-05 |

It is, I believe, interesting to notice how both the misclassifications have to do with the 'Speed limit (20 Km/h)' signal, which was originally the least represented. Also, the kind of image chosen for the 'No Entry' signal makes it probably harder to identify - it is indeed the second in the list, but with a significantly lower probability.

In both cases, I believe that, besides changing the design the network itself, further degrees of refinement on the original dataset would also help.

---

## Layer Visualization

As a final step in the project I have added a visualization for the first layer of the network, fed with the first image obtained from the web.

I believe it's interesting to notice how at this layer the network seems to "see" moslty changes in contours, with the circular boundaries of the sign getting evidenced, even if this is not yet enough to classify.

I feel this kind of analysis would prove helpful in eventual refactoring of the proposed architecture. 


---

## Conclusions, and further improvements

The design proposed satisfies the requirement of a test accuracy > 93%, and was a good exercise in experimenting with different architecture for a Convolutional Neural Network.

Among the things I would consider as further developments are:

* Reduce tendency to overfitting: droput is a possibility, but also simple pooling might be an option;
* Further refine the data set: the current augmentation strategy can be further refined, I think especially in the ways of improving brightness and contrast in the images.
