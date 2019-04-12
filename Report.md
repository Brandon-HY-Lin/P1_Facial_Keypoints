[dataset]: https://www.cs.tau.ac.il/~wolf/ytfaces/ "YouTube Faces Dataset"

[facial_keypoints_sample]: https://github.com/Brandon-HY-Lin/P1_Facial_Keypoints/blob/master/images/key_pts_example.png "Facial keypoints sample"

[architecture]: https://github.com/Brandon-HY-Lin/P1_Facial_Keypoints/blob/master/images/architecture.png "Architecture"

[result_1]: https://github.com/Brandon-HY-Lin/P1_Facial_Keypoints/blob/master/images/result_1.png "Result 1"

[result_2]: https://github.com/Brandon-HY-Lin/P1_Facial_Keypoints/blob/master/images/result_2.png "Result 2"

# Abstract
This work adopts architecture of 1 pretrained DenseNet121, 4 CNN layers, and 1 FC layer. The smooth L1 Loss of training set is 0.0136 after 36 epochs based on [YouTube faces dataset][dataset].


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
The smooth L1 Loss of training set is 0.0136 after 36 epochs. The predicted keypoints is shown below.

![Result 1][result_1]
![Result 2][result_2]


# Conclusion
In this work, an architecture with a pretrained DenseNet121, 4 conv layers, and directed path is implemented. The smooth L1 loss is 0.0136.


# Appendix
#### Hyper-Parameters

* Conv Path
    * CNN Layer 1:
        * Conv1:
            * input shape: (1, 224, 224)
            * ouput shape: (16, 224, 224)
            * kernel: 5
            * padding: 2
        * MaxPool 1:
            * stride: 2
        * Dropout 1: 0.05
    * CNN Layer 2:
        * Conv2:
            * input shape: (16, 112, 112)
            * ouput shape: (32, 112, 112)
            * kernel: 5
            * padding: 2
        * MaxPool 2:
            * stride: 2
        * Dropout 2: 0.10
    * CNN Layer 3:
        * Conv3:
            * input shape: (32, 56, 56)
            * ouput shape: (64, 56, 56)
            * kernel: 5
            * padding: 2
        * MaxPool 3:
            * stride: 2
        * Dropout 1: 0.15
    * CNN Layer 4:
        * Conv4:
            * input shape: (64, 28, 28)
            * ouput shape: (128, 24, 24)
            * kernel: 5
            * padding: 2
        * MaxPool 4:
            * stride: 2
        * Dropout 1: 0.20
    * FC 1:
        * input dim: 12*12*128
        * ouput dim: 1024

* Direct Path
    * MaxPool 1:
        * stride: 4
        * output shape: (56, 56)
    * MaxPool 2:
        * stride: 8
        * ouput shape: (28, 28)
    * MaxPool 3:
        * stride: 16 
        * output shape: (14, 14)
    * FC 2:
        * input dim: 4116 = (56x56 + 28x28 + 14x14)
        * ouput dim: 1024