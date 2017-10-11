# **Traffic Sign Recognition** 
---

- Steps of this project are the following:
  * Load the data set (see below for links to the project data set)
  * Explore, summarize and visualize the data set
  * Design, train and test a model architecture
  * Use the model to make predictions on new images
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


## Rubric Points
** Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. **

---

# Dataset Exploration
---

## CRITERIA 1 : Dataset Summary 

After loading the pickle files I got the following results:
- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

The ground truth summary that I got after loading signnames.csv:

|ClassId      |        SignName                             |
|:---------------------:|:---------------------------------:| 
|0            |                         Speed limit (20km/h)|
|1            |                         Speed limit (30km/h)|
|2            |                         Speed limit (50km/h)|
|3            |                         Speed limit (60km/h)|
|4            |                         Speed limit (70km/h)|
|5            |                         Speed limit (80km/h)|
|6            |                  End of speed limit (80km/h)|
|7            |                        Speed limit (100km/h)|
|8            |                        Speed limit (120km/h)|
|9            |                                   No passing|
|10           | No passing for vehicles over 3.5 metric tons|
|11           |        Right-of-way at the next intersection|
|12           |                                Priority road|
|13           |                                        Yield|
|14           |                                         Stop|
|15           |                                  No vehicles|
|16           |     Vehicles over 3.5 metric tons prohibited|
|17           |                                     No entry|
|18           |                              General caution|
|19           |                  Dangerous curve to the left|
|20           |                 Dangerous curve to the right|
|21           |                                 Double curve|
|22           |                                   Bumpy road|
|23           |                                Slippery road|
|24           |                    Road narrows on the right|
|25           |                                    Road work|
|26           |                              Traffic signals|
|27           |                                  Pedestrians|
|28           |                            Children crossing|
|29           |                            Bicycles crossing|
|30           |                           Beware of ice/snow|
|31           |                        Wild animals crossing|
|32           |          End of all speed and passing limits|
|33           |                             Turn right ahead|
|34           |                              Turn left ahead|
|35           |                                   Ahead only|
|36           |                         Go straight or right|
|37           |                          Go straight or left|
|38           |                                  Keep right|
|39           |                                   Keep left|
|40           |                        Roundabout mandatory|
|41           |                           End of no passing|
|42           |End of no passing by vehicles over 3.5 metric ...|


## CRITERIA 2 : Exploratory Visualization

- Below are single instance of Training images from each class : 

[image09]: ./data_visualtization/data_visualization_1.png "Traffic Sign 6"
[image10]: ./data_visualtization/data_visualization_2.png "Traffic Sign 7"
[image11]: ./data_visualtization/data_visualization_3.png "Traffic Sign 8"
[image12]: ./data_visualtization/data_visualization_4.png "Traffic Sign 9"

![Data Visualization][image09]
![Data Visualization][image10]
![Data Visualization][image11]

- Below is the Histogram Bar Chart displaying Number of samples for each sign in Training data:

![Data Visualization][image12]

# Design and Test a Model Architecture
---

## CRITERIA 1 : Preprocessing


#### Pre-processing techniques used 
----

My preprocessing pipeling has following steps :
 1. Gaussian blurring on the image
   - I have used gassian blur to smoothen the image
   - This reduces the noise from the images 
   - Used 5x5 gaussian filter on the image 
   - Used OpenCv for applying the gaussian filter
   - cv2.GaussianBlur(img, (5,5), 20.0)
 2. Crop the blurred images
   - Remove boreder pixels from the images
   - The border pixels are of no use in classification 
   - The boreder pixels are not part of the signs but the are the background in the images
   - the border pixels will confuse out model
   - So I decided to remove the border pixels
 3. Change image Contrast
   - Tried to make all the images a high contrast images 
   - This is done so that images are more clear
   - The color of the images is more solid to see by the model
 4. Equializing the histogrm of the image
   - After making th eimage high contrast there wont be much intensity variation in the similar regions of the image
   - To make a the image look more solid and clearer I applied histogram equalization 
   - This technique will improve the contrast of our images
   - Used OpenCv to Equalize the histograms
   - cv2.equalizeHist(img)
 5. Normalization
   - This is done so that all pixels are at a range from 0 to 1
   - This helps so that the high density and low density pixels don't much affect our classification
   - For example images of same class whether in brighter light or less light should not be classified as different classes
   
Below are images after pre-processing which will give a better idea of pre-processing results:

[image13]: ./pre_processing/pre_processing_1.png "Traffic Sign 10"
[image14]: ./pre_processing/pre_processing_2.png "Traffic Sign 11"
[image15]: ./pre_processing/pre_processing_3.png "Traffic Sign 12"

![Pre Processing](https://github.com/jayshah19949596/Traffic-Sign-Classifier-Project/blob/master/pre_processing/pre_processing_1.PNG)
![Pre Processing](https://github.com/jayshah19949596/Traffic-Sign-Classifier-Project/blob/master/pre_processing/pre_processing_2.PNG)
![Pre Processing](https://github.com/jayshah19949596/Traffic-Sign-Classifier-Project/blob/master/pre_processing/pre_processing_3.PNG)

## CRITERIA 2 : Model Architecture

- My model is based on Google's famous inception architecture
- I think There are many better models for classifying images than Lenet
- Google's inception module is a better architecture than Lenet so I have decided to use it 
- The model takes image of dimension 26x26x3
- The Model has follwing description:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 26x26x3 image   							    | 
| Convolution 1x1     	| 1x1 stride, SAME padding, outputs 26x26x3 	|
| RELU					|												|
| Convolution 5x5      	| 1x1 stride, SAME padding, outputs 26x26x64  	|
| RELU          	    |                                               |
| Inception-1			| outputs 26x26x256   							|
| Max pooling 2x2      	| 2x2 stride, outputs 13x13x256				    |
| Inception-2    	    | outputs 13x13x512                             |
| Max pooling 3x3		| 2x2 stride, outputs 6x6x512    			    |
| Convolution    		| outputs 6x6x256								|
| Fully connected		| input 9216, output 512        				|
| RELU					|												|
| Dropout				| 75% keep        								|
| Fully connected		| input 512, output 43     						|

- To know more about Google's Inception architecture please refer the paper [GoogleNet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf).


## CRITERIA 3 : Model Training

- I have trained the model for 10 epochs
- I tried training for 30 epochs but it seems the method I have used for pre-processing and model used there is not much changes in the validation accuracy when number of epochs is 10 or 30
- I have used bacth size of 128. I have choosen this because tensorflow gives optimized results when batch size is a power of 2
- learning rate used is 0.001. I think this is ideal leanring rate to start with or may be 0.01
- Used AdamOptimizer to optimize cost function.
- AdamOptimizer is better than other optimizers. It helps in adaptive leanring and is useful for large data with model having lot's of hyperparmaters and anyways it was default optimizer when I tried Lenet on The traffic sign Dataset


## CRITERIA 4 : Solution Approach

My final model results were:
* Validation set accuracy of my model is 93.7 
* Test set accuracy of my model is 93.5

Iterative approach was chosen to achieve the solution :
* First I just played around with LeNet
* LeNet gave an accuracy of 89
* I then normalized my data and converted them to greyscale and saw that model still gives an accuracy of 89 for 10 epochs
* I increased the number of epochs to 30 and I achieved an accuracy of 91.5
* I again ran the model without any changes and I achieved an accuracy of 92.9
* So then I decided to add dropouts without making any other changes in my solution
* Even adding dopouts did not help me reach an accuracy of 93
* I decided to change my solution
* All my above solution was before applying data visualization
* When I did data visualization, I found that data augmentation is necessary for my model to perform better 
* And then I decided the current solution i.e Using inception architecture and a more sophasticated pre-processing pipeline as mentioned above
* Now with my new solution I get an accuracy of 93% in the first epoch 
* I did not want to try any other solution because it is very time consuming to run the network on my device 
* So I settled down with this solution
* I think if I convert the images to greyscale then my model will perform even better... but I am not sure about this
* I choose inception architecture because it is better than LeNet as it gives bettwer results than LeNet

## CRITERIA 1 : Acquiring New Images


- The custom test images were given to me by my Mentor
- I think they are taken from the web... I am not sure...!!! 
- There are 8 custom images
- The first 3 images should be correctly calssified by the model because the image quality is good and they are very straight forward as there are no other class images that has similarity to the first 3 images... This is what I think
- The next 3 images have a lot's of similarity with data from other classes... Model should have a very low confidence on classifying these images ... by low confidence I mean that the probability of the softmax of the class predicted will be less even thought it's the maximum for that prediction but compared to other image prediction the softmax should be less
- So the last two images are a bit blurry and it should be difficult for the model to classify the last two images... I think the last two images should be misclassified by the model

- Below Are the Images choosen:

[image16]: ./custom_data/02.jpg
[image17]: ./custom_data/04.jpg 
[image18]: ./custom_data/14.jpg
[image19]: ./custom_data/18.jpg
[image20]: ./custom_data/19.jpg
[image21]: ./custom_data/25.jpg
[image22]: ./custom_data/38.jpg
[image23]: ./custom_data/40.jpg

![Test Image][image16]
![Test Image][image17]
![Test Image][image18]
![Test Image][image19]
![Test Image][image20]
![Test Image][image21]
![Test Image][image22]
![Test Image][image23]


## CRITERIA 2 : Performance on New Images

- When I checked the performance on my model on New Images I got an accuracy of 62.5
- Which means out of 8 my model classified 5 images correctly
- I was happy with my model's performance
- I wish I could have greyscaled the images and tried checking out the performance but I am constrainted with time and my machine is not working properly
- As Expected the model Misclassified the last two images in the above custom images shown
- Below are the Prediction Result of my model


[image24]: ./custom_test/test_prediction.PNG
![Prediction][image24]

- The summary of the prediction is as follows:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution      		   | Slippery Road   					     		| 
| Speed Limit(50 km/hr)		   | Speed Limit(50 km/hr)   		     			| 
| Dangerous curve to the left  | Dangerous curve to the left     				| 
| Stop     			           | Stop    										|
| keep right			       | Speed Limit(80 km/hr)							|
| Road Work     		       | Road Work   				 	     			|
| Roundabout Mandatory	       | Slippery Road      							|
| Speed Limit(70 km/hr)        | Speed Limit(70 km/hr)      					| 


## CRITERIA 3 : Performance on New Images

- The top five softmax probabilities of the predictions on the captured images are outputted.
- Below are my softmax results
- The probability scores of the first image does makes sense even though none of them is right
- The second probability score of second last image is correct
- All the other clear images are perfectly classified
- You can see that correctly classified images are classified with a score of almost 1.0 which is really great 

[image25]: ./custom_test/test_prediction_1.PNG
[image26]: ./custom_test/test_prediction_2.PNG

![Prediction][image25]
![Prediction][image26]


## (Optional 1) AUGMENT THE TRAINING DATA
 
- After doing data visualisation on our raw data I was sure that data sugmentation is necessary for our model to perform better
- The data provided for training is not enough 
- Many classes of our training data has less number for training data.
- It is possible that there is a chance of bias training as many classes has lot's of training data while many classes have less training data
- Because of less data for few classes the model will give prediction of the data for the class it has seen the most
- This is what I thought of and decided that there has to be data augmentation needed
- The results for augmented data i.e. on instance for each class is displayed below:

[image27]: ./data_augmentation/data_augmentation_1.PNG
[image28]: ./data_augmentation/data_augmentation_2.PNG

![Prediction][image27]
![Prediction][image28]
