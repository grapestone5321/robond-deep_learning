# Writeup

## 1. Introduction

### Deep Learning Project ##

In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

### Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us).
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation! 

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

We can avoid that by using 1x1 convolutions.

1x1 convolution helped in reducing the dimensionality of the layer. A fully-connected layer of the same size would result in the same number of features. However, replacement of fully connected layers with convolutional layers presents an added advantage that during inference (testing your model), you can feed images of any size into your trained network.

Replacing a fully connected layer with 1x1 convolutional layers will result in the output value with tensor will remain 4D instead of flattening to 2D, so spatial information will be preserved.

The output of the convolution operation is the result of sweeping the kernel over the input with the sliding window and performing element wise multiplication and summation.

The number of kernels is equivalent to the number of outputs in a fully connected layer. Similarly, the number of weights in each kernel is equivalent to the number of inputs in the fully connected layer. Effectively, this turns convolutions into a matrix multiplication with spatial information.

### Separable Convolutions
The Encoder for our FCN essentially require separable convolution layers. The 1x1 convolution layer in the FCN, however, is a regular convolution. Implementations for both are provided for our use. Each includes batch normalization with the ReLU activation function applied to the layers.

### Bilinear Upsampling
The helper function implements the bilinear upsampling layer. Upsampling by a factor of 2 is generally recommended. Upsampling is used in the decoder block of the FCN.

## 4. Build the Model 

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

### Hyperparameters
Hyperparameters are defined and tuned.
- batch_size: number of training samples/images that get propagated through the network in a single pass.
- num_epochs: number of times the entire training dataset gets propagated through the network.
- steps_per_epoch: number of batches of training images that go through the network in 1 epoch. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.

     steps_per_epoch = number_of_ training_images/batch_size

- validation_steps: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset.

     validation_steps = number_of_validation_images/batch_size

- workers: maximum number of processes to spin up. This can affect our training speed and is dependent on our hardware. 

```
learning_rate = 0.005
batch_size = 16
num_epochs = 20
steps_per_epoch = 300
validation_steps = 100
workers = 4
```

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

### Scoring
To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask.

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

### The neural network achieves a minimum level of accuracy for the network implemented: 
In the case of batch_size=32 and num_epochs=20, the final score is 37%.

In the case of batch_size=16 and num_epochs=10, the final score is 39%.

My final score is 43% (batch_size=16, num_epochs=20). The accuracy greater than or equal to 40% (0.40) is obtained using the Intersection over Union (IoU) metric.

### Limitations to the neural network with the given data chosen for various follow-me scenarios. 
This model and data would not work well for following another object (dog, cat, car, etc.) instead of a human. Some changes would be required.
- Additional data would be required to discriminate new object from others, and the network needs to train again using those images. 
- Add an additional output node for this new object, and add labels to training data with this label.
