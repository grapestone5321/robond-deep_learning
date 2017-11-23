# Writeup

## Introduction

### Deep Learning Project ##

In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

### Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us).
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation! 

## 1. Data Collection
A starting dataset has been provided for this project. Additional data of my own are also collected to improve my model. The total number of images in training/validation dataset are as follows:

number_of_ training_images = 5075

number_of_validation_images = 1896

## 2. FCN Layers 
In the Classroom, we discussed the different layers that constitute a fully convolutional network (FCN). We need the functions to build our semantic segmentation model.

### Separable Convolutions
The Encoder for our FCN essentially require separable convolution layers. The 1x1 convolution layer in the FCN, however, is a regular convolution. Implementations for both are provided for our use. Each includes batch normalization with the ReLU activation function applied to the layers.

### Bilinear Upsampling
The helper function implements the bilinear upsampling layer. Upsampling by a factor of 2 is generally recommended. Upsampling is used in the decoder block of the FCN.

## 3. Build the Model 

![deep_learning|FCN](https://cldup.com/wbwKbPM8uJ.png)

An FCN is built to train a model to detect and locate the hero target within an image. The steps are:
- Create an encoder_block 
- Create a decoder_block 
- Build the FCN consisting of encoder block(s), a 1x1 convolution, and decoder block(s). This step requires experimentation with different numbers of layers and filter sizes to build your model.

### Encoder Block
An encoder block includes a separable convolution layer using the separable_conv2d_batchnorm() function. The filters parameter defines the size or depth of the output layer. For example, 32 or 64.

### Decoder Block
The decoder block is comprised of three parts:
- A bilinear upsampling layer using the upsample_bilinear() function. The current recommended factor for upsampling is set to 2.
- A layer concatenation step. This step is similar to skip connections. I concatenate the upsampled small_ip_layer and the large_ip_layer.
- Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.

### Model
My FCN architecture is built.

There are three steps:
- Add encoder blocks to build the encoder layers.
- Add a 1x1 Convolution layer using the conv2d_batchnorm() function. 1x1 Convolutions require a kernel and stride of 1.
- Add decoder blocks for the decoder layers.

I used three layers for encoder and decoder. Three layers seem to be sufficient to obtain the target score.

```
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

## 4. Training 
The FCN is used and an ouput layer is defined based on the size of the processed image and the number of classes recognized. The hyperparameters are defined to compile and train your model.

### Hyperparameters
Hyperparameters are defined and tuned.
- batch_size: number of training samples/images that get propagated through the network in a single pass.
- num_epochs: number of times the entire training dataset gets propagated through the network.
- steps_per_epoch: number of batches of training images that go through the network in 1 epoch. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.

     steps_per_epoch = number_of_ training_images/batch_size

- validation_steps: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset.

     validation_steps = number_of_validation_images/batch_size

- workers: maximum number of processes to spin up. This can affect our training speed and is dependent on our hardware. 

### training curves:
![deep_learning|training](https://cldup.com/wbwKbPM8uJ.png)

## 5. Prediction 
Now that I have my model trained and saved, I make predictions on my validation dataset. These predictions can be compared to the mask images, which are the ground truth labels, to evaluate how well my model is doing under different conditions.

There are three different predictions available from the helper code provided:
- patrol_with_targ: Test how well the network can detect the hero from a distance.
- patrol_non_targ: Test how often the network makes a mistake and identifies the wrong person as the target.
- following_images: Test how well the network can identify the target while following them.

The predictions are written to files and return paths to the appropriate directories. Now lets look at my predictions, and compare them to the ground truth labels and original images. Some sample images are visualized from the predictions in the validation set.

### images while following the target:
![deep_learning|FCN](https://cldup.com/wbwKbPM8uJ.png)
### images while at patrol without target:
![deep_learning|FCN](https://cldup.com/wbwKbPM8uJ.png)
### images while at patrol with target:
![deep_learning|FCN](https://cldup.com/wbwKbPM8uJ.png)

## 6. Evaluation 
My model is evaluated. Several different scores are included to evaluate my model under the different conditions discussed during the Prediction step.

### Scoring
To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask.

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

### The neural network achieves a minimum level of accuracy for the network implemented: 
My final score is 43%. The accuracy greater than or equal to 40% (0.40) is obtained using the Intersection over Union (IoU) metric.


### 1. Provide a write-up document including all rubric items addressed in a clear and concise manner.

- The write-up should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled. The write-up should include a discussion of what worked, what didn't and how the project implementation could be improved going forward.

- This report should be written with a technical emphasis (i.e. concrete, supporting information and no 'hand-waiving'). Specifications are met if a reader would be able to replicate what you have done based on what was submitted in the report. This means all network architecture should be explained, parameters should be explicitly stated with factual justifications, and plots / graphs are used where possible to further enhance understanding. A discussion on potential improvements to the project submission should also be included for future enhancements to the network / parameters that could be used to increase accuracy, efficiency, etc. It is not required to make such enhancements, but these enhancements should be explicitly stated in its own section titled "Future Enhancements". 

### 2. The write-up conveys the an understanding of the network architecture. 

- The student clearly explains each layer of the network architecture and the role that it plays in the overall network. The student can demonstrate the benefits and/or drawbacks of different network architectures pertaining to this project and can justify the current network with factual data. Any choice of configurable parameters should also be explained in the network architecture. 

- The student shall also provide a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.

### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network. 

- The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

     Epoch. Learning Rate. Batch Size. Etc. 

- All configurable parameters should be explicitly stated and justified. 

### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

- The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used. 

- The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.

### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

- The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up. 

- The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required. 

### Model

### 7. The model is submitted in the correct format.

- The file is in the correct format (.h5) and runs without errors.

### 8. The neural network must achieve a minimum level of accuracy for the network implemented. 

- The neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric.
