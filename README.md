# oxidized_nn
The purpose of this project was to help develop my understanding of neural networks and deep learning. All of the code is based on code provided by Institute of Advanced Analytics (NC State) for our Deep Learning course. While the neural network created by this project is very rudimentary, consisting of only five layers with no convolution or max pooling, I think it was definitely helpful in helping me understand the concepts involved a bit better. 

From start to finish, the program performs the following steps:
1. Read in the `wheat.csv` file
1. Normalize the dataset using the min and max values from each column
1. Create the neutral network
1. Create a 5-fold cross-validation split on the dataset
1. Train the network, calculating the accuracy of the network's predictions for each fold
1. Calculate the mean accuracy from each of the five folds

The training consists of five steps:
1. Forward propagate the training data row through the network.
1. One hot encode expected output classification.
1. Backpropagate error through network to get error gradient (delta).
1. Update weights based on error gradient.
1. Decay learning rate.