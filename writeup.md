#**Traffic Sign Recognition - Project#2 - CarND** 

##Project Writeup

---

**Build a Traffic Sign Recognition Project**

The goals of this project are the following:
* Load the GTSRB training and test dataset (http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=dataset)  
* Explore, summarize and visualize the data set
* Design, train and test a model architecture using TensorFlow
* Use the model to make predictions on new images, downloaded from Internet.
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./snapshots/tableSigns.png "Table of dataset"
[image10]: ./snapshots/training_dataset_visualization.png "train_visual"
[image11]: ./snapshots/afterPreprocessing.png "afterPreProcessing_visual"

## Main Submission files
###Here is the [HTML of IPython Notebook](https://github.com/vkrishnam/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.html) which captures the snapshot of the all code cells being executed and their results.  

---
###GitHub Repo


Here is a link to [project code](https://github.com/vkrishnam/Traffic_Sign_Classifier/)

###Data Set Summary & Exploration

####1. The Traffic sign dataset is loaded into the project using pickle module by reading the .p files.

The code for this step is contained in the first few code cells of the IPython notebook.  

Used numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. An exploratory visualization of the dataset.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a table showing how the dataset is spread.

Some points to note:

* The images in the dataset are either too dark or too bright!!! 
* Majority of the images are triangles and circles. Real distintion among them is the details inside those traingles and circles.
* The background of the images has varied content.

![Table of Signs][image9]


![alt text][image10]




###Design and Test a Model Architecture

####1. Pre-Processing:

The code for this step is contained in the fourth code cell of the IPython notebook.

- As a first step, Image normalization is tried out, as we noted while visualizing the dataset that the images are either too bright or too dark. Even here two types of normalization are tried out. 

1. Linear scaling and 
2. Histogram equalization. 

The second method is superior as it spreads the distribution uniformly across the range.

- As a second step, convert the image to YUV color space/domain as majority of the information is Luma and even to seperate sign on color basis UV would be better.

- As a last and third step we find a mean image among all the training dataset, and remove mean image from the images. This step is more called Mean normalization is to remove the variations of different backgrounds, lightings etc.

Here is an snapshot of different traffic sign images after pre-processing.

![alt text][image11]


####2. Setting up training, validation and testing data. 

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

As the dataset originally do not have validation set, one such validation set had to be created for analysing the progress of optimization during training process.
While the test dataset is untouched, we observe that different labels among the 43 signs have varying amount of training samples in the training dataset.
As we wanted a training set which has good representation of each of the classes, we went ahead with choosing 20% of training samples of each of the class as validation dataset.

So final Training set had 31368 number of images. 
Validation set and Test set had 7841 and 12630 number of images respectively.


####3. Model architecture 

To start with, used the LeNet Model architecture used for the MNIST except for the input layer changed to the 32x32x3 instead of 28x28x3.
This LeNet model trained for 10 epochs and with learning rate of 0.001 could achieve a validation accurary of ~0.90.
But accuracy could not go beyond 0.90.

Then a modified LeNet architecture named TscNet which is described as follows is tried out.

The code for final model is located in the sixth cell of the ipython notebook. 

Final model consisted of the following layers:

|:---------------------:|:---------------------------------------------:| 
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 YUV image   							| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x3 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x6  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 11x11x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6    				|
| Fully connected		| Input 400, Output 120							|
| RELU					|												|
| Fully connected		| Input 120, Output 84							|
| RELU					|												|
| Fully connected		| Input 84, Output 10							|
| SoftMax				| Output 10 probs								|
|						|												|
|:---------------------:|:---------------------------------------------:| 
 
The initial conv layer of 1x1 is motivated by the Inception or NiN concept to find the latent features.
Also the later on Conv layers filter sizes are gradually increased from 3x3 to 5x5 to increase the receptive field at each subsequent layer.


####4. Training the Model

Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the senventh cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
