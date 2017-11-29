# Writeup

## 1. Introduction

### Deep Learning Project ##
In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

### Semantic segmentation
Semantic segmentation is the task of assigning meaning to part of an object. This can be done at the pixel level where we assign each pixel to a target class such as road, car, pedestrian, sign, or any number of other classes. Semantic segmentation helps us derive valuable information about every pixel in the image rather than just slicing sections into bounding boxes. This is a field known as scene understanding and it's particulaly relevant to autonomous vehicles. Full scene understanding help with perception, which enables vehicles to make decisions.

## 2. Data Collection
A starting dataset has been provided for this project. Additional data of my own are also collected to improve my model. The total number of images in training/validation dataset are as follows:

number_of_ training_images = 5075

number_of_validation_images = 1896

## 3. FCN Layers 
In the Classroom, we discussed the different layers that constitute a fully convolutional network (FCN). We need the functions to build our semantic segmentation model.

### Fully convolutional networks (FCNs)
While doing the convolution, they preserve the spatial information throughout the entire network.

### 1x1 Convolutions
We covered how, in TensorFlow, the output shape of a convolutional layer is a 4D tensor. However, when we wish to feed the output of a convolutional layer into a fully connected layer, we flatten it into a 2D tensor. This results in the loss of spatial information, because no information about the location of the pixels is preserved.

Replacing a fully connected layer with 1x1 convolutional layers will result in the output value with tensor will remain 4D instead of flattening to 2D, so spatial information will be preserved.

The output of the convolution operation is the result of sweeping the kernel over the input with the sliding window and performing element wise multiplication and summation. The number of kernels is equivalent to the number of outputs in a fully connected layer. Similarly, the number of weights in each kernel is equivalent to the number of inputs in the fully connected layer. Effectively, this turns convolutions into a matrix multiplication with spatial information.

## 4. Build the Model 

![deep_learning|FCN](https://cldup.com/wbwKbPM8uJ.png)

An FCN is built to train a model to detect and locate the hero target within an image. The steps are:
- Create an encoder_block 
- Create a decoder_block 
- Build the FCN consisting of encoder block(s), a 1x1 convolution, and decoder block(s). This step requires experimentation with different numbers of layers and filter sizes to build your model.

### Encoder Block
An encoder block includes a separable convolution layer using the separable_conv2d_batchnorm() function. The filters parameter defines the size or depth of the output layer. For example, 32 or 64.

### Separable Convolutions
The Encoder for our FCN essentially require separable convolution layers. The 1x1 convolution layer in the FCN, however, is a regular convolution. Implementations for both are provided for our use. Each includes batch normalization with the ReLU activation function applied to the layers.

### Decoder Block
The decoder block is comprised of three parts:
- A bilinear upsampling layer using the upsample_bilinear() function. The current recommended factor for upsampling is set to 2.
- A layer concatenation step. This step is similar to skip connections. I concatenate the upsampled small_ip_layer and the large_ip_layer.

- Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.

### Bilinear Upsampling
The helper function implements the bilinear upsampling layer. Upsampling by a factor of 2 is generally recommended. Upsampling is used in the decoder block of the FCN.

Transposed convolutions are one way of upsampling layers to higher dimensions or resolutions. Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. The weighted average is usually distance dependent. Let's consider the scenario where you have 4 known pixel values, so essentially a 2x2 grayscale image. This image is required to be upsampled to a 4x4 image. The bilinear upsampling method will try to fill out all the remaining pixel values via interpolation.

### Skip connections
One effect of convolutions or encoding in general is you narrow down the scope by looking closely at some picture and lose the bigger as a result. So even if we were to decode the output of the encoder back to the original image size, some information has been lost.

Skip connection are a way of retraining the information easily. The way skip conection work is by connectiog the output of one layer to a non-ajacent layer. The output of the pooling layer from the encoders combine with the current layers output using the element-wise addition operation. The result is bent into the next layer. These skip connections allow the network to use information from multiple resolutions. As a result, the network is able to make more precise segmentation decisions.

### Model
My FCN architecture is built.

There are three steps:
- Add encoder blocks to build the encoder layers.
- Add a 1x1 Convolution layer using the conv2d_batchnorm() function. 1x1 Convolutions require a kernel and stride of 1.
- Add decoder blocks for the decoder layers.

![deep_learning|FCN](https://cldup.com/wbwKbPM8uJ.png)

I used three layers for encoder and decoder. Three layers seem to be sufficient to obtain the target score.

```
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_layer1 = encoder_block(inputs, filters=32, strides=2)
    encoder_layer2 = encoder_block(encoder_layer1, filters=64, strides=2)
    encoder_layer3 = encoder_block(encoder_layer2, filters=128, strides=2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    mid_layer = conv2d_batchnorm(encoder_layer3, filters=8, kernel_size=1, strides=1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_layer1 = decoder_block(mid_layer, encoder_layer2, filters=128)
    decoder_layer2 = decoder_block(decoder_layer1, encoder_layer1, filters=64)
    decoder_layer3 = decoder_block(decoder_layer2, inputs, filters=32)   
    
    x = decoder_layer3    
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

## 5. Training 
The FCN is used and an ouput layer is defined based on the size of the processed image and the number of classes recognized. The hyperparameters are defined to compile and train your model.
labels to training data with this label.
### Hyperparameters
Hyperparameters are defined and tuned. They are optimized via trial and error.
- batch_size: number of training samples/images that get propagated through the network in a single pass.
- num_epochs: number of times the entire training dataset gets propagated through the network.     
- steps_per_epoch: number of batches of training images that go through the network in 1 epoch. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.
- validation_steps: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset.
labels to training data with this label.
- workers: maximum number of processes to spin up. This can affect our training speed and is dependent on our hardware. 

```
learning_rate = 0.005
batch_size = 16
num_epochs = 20
steps_per_epoch = 300
validation_steps = 100
workers = 4
```

Learning rate 0.005 is more stable than 0.05 and improves the accuracy faster than 0.0005. Batch size 16 obtains the higher final score in comparison to Batch size 32. Epoch needs 20 iterations to get enough accuracy.

steps_per_epoch = number_of_ training_images/batch_size

validation_steps = number_of_validation_images/batch_size

### training curves:
![deep_learning|training](https://cldup.com/seigdzIv9O.png)
## 6. Prediction 
Now that I have my model trained and saved, I make predictions on my validation dataset. These predictions can be compared to the mask images, which are the ground truth labels, to evaluate how well my model is doing under different conditions.

There are three different predictions available from the helper code provided:
- patrol_with_targ: Test how well the network can detect the hero from a distance.
- patrol_non_targ: Test how often the network makes a mistake and identifies the wrong person as the target.
- following_images: Test how well the network can identify the target while following them.

The predictions are written to files and return paths to the appropriate directories. Now lets look at my predictions, and compare them to the ground truth labels and original images. Some sample images are visualized from the predictions in the validation set.

### images while following the target:
![deep_learning|following_the_target](https://cldup.com/rHDoHE4DeT.png)
### images while at patrol without target:
![deep_learning|patrol_without_target](https://cldup.com/92PMxe5Jn0.png)
### images while at patrol with target:
![deep_learning|patrol_with_target](https://cldup.com/JnQmqmJ7Gb.png)

## 7. Evaluation 
My model is evaluated. Several different scores are included to evaluate my model under the different conditions discussed during the Prediction step.

### How the Final score is Calculated
The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

### The neural network achieves a minimum level of accuracy for the network implemented: 
In the case of batch_size=32 and num_epochs=20, the final score is 37%.

In the case of batch_size=16 and num_epochs=10, the final score is 39%.

My final score is 43% (batch_size=16, num_epochs=20). The accuracy greater than or equal to 40% (0.40) is obtained using the Intersection over Union (IoU) metric.

### Limitations to the neural network with the given data chosen for various follow-me scenarios. 
This model and data would not work well for following another object (dog, cat, car, etc.) instead of a human. Some changes would be required.
- Add data with labels to discriminate new object from others. 
- Train the network again using those images.

## References
[1] Udacity, Robotics Nanodegree Program, Part 6 Deep Learning.

[2] https://github.com/udacity/RoboND-DeepLearning-Project
