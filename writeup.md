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
[image4]: ./new_images/120px-Zeichen_131svg_Traffic_signals.png "Traffic Sign 1"
[image5]: ./new_images/Zeichen_112_–_Unebene_Fahrbahn,_StVO_1970svg_Bumpy_road.png "Traffic Sign 2"
[image6]: ./new_images/Zeichen_136-10_-_Kinder,_Aufstellung_rechts,_StVO_1992svg_ChildrenCrossing.png "Traffic Sign 3"
[image7]: ./new_images/Zeichen_206svg_Stop.png "Traffic Sign 4"
[image8]: ./new_images/Zeichen_267svg_NoEntry.png "Traffic Sign 5"
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
| Concatenation         | Inputs 400, 1350, Output 1750                 |
| Fully connected		| Input 1750, Output 120						|
| RELU					|												|
| Fully connected		| Input 120, Output 84							|
| RELU					|												|
| Fully connected		| Input 84, Output 10							|
| SoftMax				| Output 10 probs								|
|						|												|
|:---------------------:|:---------------------------------------------:| 
 
The initial conv layer of 1x1 is motivated by the Inception or NiN (Network in Network) concept to find the latent features.
Also the later on Conv layers filter sizes are gradually increased from 3x3 to 5x5 to increase the receptive field at each subsequent layer.
The result of the 3x3 convolution and 5x5 convolutions are concatenated to feed into Fully Connected layer, thats becuase the 3x3 activation might capture the very low level detail which might be useful for the end classification.
Thats motivated from the reading of the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).



####4. Training the Model

Training is done with SoftMax Cross Entropy as the loss function and Adam Optimizer is used over SGD (Stocastic Gradient Descent).  Batch size of 128, 20 epochs and (hyperparameters) learning rate of 0.001 is choosen.

The code for training the model is located in the seventh cell of the ipython notebook. 


####5. Approach taken for finding a solution. 

Final solution is arrived as a iterative process.

First the effort was put into getting the pipeline (Dataset split, Model, Training and Validation Accuracy) up without any Pre-Processing of dataset and LeNet as the Model Architecture.
This model was not able to learn much as the accuracy cannot got beyond ~0.85. Also training accuracies where quite low.
It is understood that the training dataset of different signs are indistinguishable for humans itself. That highlighted the need for pre-processing.

Then the pre-processing steps (histogram equalization, mean normalization) are implemented to improve the accuracies further.

The TscNet model architecture is worked out upon reading and borrowing ideas from different papers.
Also analyzing the dataset, it became clear that the shapes have got significance in classifing the sign than than the color. So the YUV data is feed to the network rather than RGB.
The initial conv layer of 1x1 is motivated by the Inception or NiN (Network in Network) concept to find the latent features.
Also the later on Conv layers filter sizes are gradually increased from 3x3 to 5x5 to increase the receptive field at each subsequent layer.
The result of the 3x3 convolution and 5x5 convolutions are concatenated to feed into Fully Connected layer, thats becuase the 3x3 activation might capture the very low level detail which might be useful for the end classification.
Thats motivated from the reading of the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

The code for calculating the accuracy of the model is located in the eigth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.970
* test set accuracy of 0.898

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that were found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images might be difficult to classify because the stop sign, bumpy road sign and traffic light sign might need the detailed edge information and also there are very less of those training samples compared to other signs in the training dataset.

####2. The model's predictions on new traffic signs and compare the results to predicting on the test set. 


The code for making predictions on my final model is located in the ninth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic Signals      	| Traffic Signals 								| 
| Bumpy road     		| Bumpy road									|
| Children crossing		| Children crossing								|
| Stop   	      		| Speed limit (30km/h)			 				|
| No Entry	    		| No Entry          							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 
This compares favorably to the accuracy on the test set of ~0.898
This is also reflected in the prediction accuracy for those classes of images in the test set.
Predctions accuracy in testSet for label 26  is : 0.789
Predctions accuracy in testSet for label 22  is : 0.850
Predctions accuracy in testSet for label 28  is : 0.773
Predctions accuracy in testSet for label 14  is : 0.970
Predctions accuracy in testSet for label 17  is : 0.975

####3. How certain the model is when predicting on each of the five new images:

By looking at the softmax probabilities for each prediction, we can say the model is predicting with high confidence as the probabilities of the predictions are in the high 0.9 range.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Traffic signal (probability of 0.999), and the image does contain a Traffic signals. 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999        			| Traffic Signals								| 
| .00005   				| Pedestrians									|


For the second image ...
#############################################################
newImage#     0 : Actual label is  26
  Top5 predictions:    [26 27 18 24 30 ]
  Top5 probabilities:  [  9.99935031e-01   5.99802770e-05   5.02341345e-06   5.95297908e-11
   1.92929752e-11]
#############################################################
newImage#     1 : Actual label is  22
  Top5 predictions:    [22 29 25 24 15]
  Top5 probabilities:  [  9.99999046e-01   8.80185326e-07   7.91025272e-08   5.18264154e-10
   1.54535603e-11]
#############################################################
newImage#     2 : Actual label is  28
  Top5 predictions:    [28 30 29 25 27 ]
  Top5 probabilities:  [  9.99999881e-01   1.11182516e-07   1.33085996e-08   2.94230729e-09
   6.94131363e-10]
#############################################################
newImage#     3 : Actual label is  14
  Top5 predictions:    [ 1 14  3 10  4]
  Top5 probabilities:  [  9.99981880e-01   1.52240200e-05   2.86367231e-06   4.90745862e-08
   2.41365883e-09]
#############################################################
newImage#     4 : Actual label is  17
  Top5 predictions:    [17 14 22 32 34]
  Top5 probabilities:  [  1.00000000e+00   3.67149226e-15   7.48519839e-16   1.07239275e-16
   4.62220254e-17]
#############################################################
