# Traffic Sign Recognition Writeup


The goals / steps of this project are the following:

* Load the dataset
* Explore, summarize and visualize the dataset
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
 

---
Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

## Dataset Summary & Exploration
The dataset used in this project is a pickled subset of the Germen Traffic Sign Detection Benchmark Dataset

### 1. Statistical Summary

By simply reading the dimensions of the loaded pickle, the statistical summary of the traffic signs dataset is as follows:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is `32*32*3`
* The number of unique classes/labels in the dataset is 43

### 2. Class Distribution Visualization

This is a bar chart showing the distribution of the data by unique classes. We can see that the distribution is very uneven with some classes contain more than 1500, while some other classes contain less than 300 samples. Hence intuitively, argumenting these minority classes will improve the performance of the model.

![alt text][image1]

### 3. Data Augumentation

To augment the data, I design the following method using `scipy.ndimage` to either randomly translate the image within [-2, 2] pixels or rotate the image with a random angle within [-10, 10] degrees.

```python
from scipy.ndimage import interpolation as ip

def augment(img):
    if (random.choice([True, False])):
        img = ip.shift(img, 
                       [random.randrange(-2, 2), 
                        random.randrange(-2, 2), 
                        0])
    else:
        img = ip.rotate(img, 
                        random.randrange(-10, 10), 
                        reshape=False)
    return img
```

In practice, all classes will be augmented to achieve 2000 samples, i.e. multiplying by the quotient of 2000 dividing by the number samples

**NOTE:** the original training set and validation set were merged before the augmentation, a new validation set will be splited from the augmented dataset using `train_test_split` method from `scikit-learn`

### 4. Summary after Augmentation
After augmentation, the summary of the dataset is as follows:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?

The distribution by class of the training set is as follows:

## Design, Train and Evaluate the Model

### 1. Pre-processing

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the dataset, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original dataset and the augmented dataset is the following ... 


### 2. Network Structure

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
 


### 3. Trainging

To train the model, I used an ....

### 4. Evaluation

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
 

## Testing on New Images

### 1. Test Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

### 2. Test Results

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

### 3. Discussion on Performance

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

## Postscipt
