[dataset]: https://www.cs.tau.ac.il/~wolf/ytfaces/ "YouTube Faces Dataset"

[facial_keypoints_sample]: https://github.com/Brandon-HY-Lin/P1_Facial_Keypoints/blob/master/images/key_pts_example.png "Facial keypoints sample"

[architecture]: https://github.com/Brandon-HY-Lin/P1_Facial_Keypoints/blob/master/images/architecture.png "Architecture"

# Abstract
This work adopts architecture of 1 pretrained DenseNet121, 4 CNN layers, and 1 FC layer. The smooth L1 Loss of training set is 0.0136 after 36 epochs. 


# Introduction
Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Some examples of these keypoints are pictured below.

![Facial keypoints sample][facial_keypoints_sample]


# Implementation
At first time, I use 6 conv layers, but it converges too slowly. Then I reduce the number of conv to 4. And I also concate downsample input to the 2nd fully-connected layer. In short the architecture is 4 conv + 3 FC. To avoid overfitting, I add dropout layer at each layer.

![architecture][architecture]


For loss function, I choose SmoothL1 which tightly fits to labels. Compared with MSE loss, SmoothL1 converges faster. Batch normalization is also added to speed training at first few epochs. Aside from the above architectures, I also tried following architectures which are bad or hard to converge.

    * Transfer learning of DensNet121
    * Using pretrained DensNet121
    * Concatenate downsampled input with conv2 or conv3.


# Results
The number of test images is 40,504. The training time for 3 epochs is 8 hours. The training loss is 1.93 and BLEU-4 of testing dataset is 0.517.


# Conclusion
In this work, CNN-RNN model is implemented and it achieves BLEU-4 score of 0.517.


# Future Works
Visualize attention by implementing [Xu's work](https://arxiv.org/pdf/1502.03044.pdf).


# Appendix
#### Hyper-Parameters

* Encoder
	* CNN
		* pretrained ResNet50 provided by pytorch
            * input size: (224, 224, 1;)
        * Linear:
            * output size: 256
* Decoder
    * Embedding:
        * input size: 9955    (vocabulary size)
        * output size: 512
	* LSTM
		* #layer: 1
        * input size: 512
	    * hidden size: 512
	* Fully-connected layer
		* layer: 1
		* input size: 512
		* output size: 9955   (vocabulary size)