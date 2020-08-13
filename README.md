# Image classification techniques

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manuelemacchia/image-classification-techniques/blob/master/notebook.ipynb)

Evaluation of transfer learning and data augmentation in AlexNet and ResNet for image classification on Caltech-101. Experiments are carried out in PyTorch. More information is available in the [PDF report]([report.pdf]).

## Structure
- `notebook.ipynb` contains the main logic of the program and the results of the experiments that we carried out.

- The `data` folder contains the Caltech-101 dataset along with train and test indices, `train.txt` and `test.txt`, as well as the dataset handler class `dataset.py`, with methods to initialize and handle the dataset and to perform a stratified split for training and validation sets.

- `manager.py` contains the network manager class, which handles training, validation and testing of a neural network, as well as logging results such as training and validation loss and accuracy over epochs.

## References
[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. _ImageNet classification
with deep convolutional neural networks._ Communications of the ACM, 2017.

[2] Li Fei-Fei, Rob Fergus, and Pietro Perona. _Learning Generative Visual Models from
Few Training Examples: An Incremental Bayesian Approach Tested on 101 Object
Categories._ IEEE Computer Society Conference on Computer Vision and Pattern Recognition
Workshops, 2004.

[3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. _ImageNet: A
large-scale hierarchical image database._ In 2009 IEEE Conference on Computer Vision
and Pattern Recognition, 2009.

[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. _Deep Residual Learning
for Image Recognition._ 2015.
