# E2E
End-to-End Self-Driving via Convolutional Neural Networks

This Code is a piece from my machine learning lab assignment that involves implementing a Convolutional Neural Network (ConvNet) for end-to-end self-driving cars. The implementation follows the architecture proposed in the paper titled "End to End Learning for Self-Driving Cars."

Resources for understanding ConvNets:
   
    a. https://www.ismll.uni-hildesheim.de/lehre/dl-20s/script/dl-05-cnn.pdf
    b. https://cs231n.github.io/convolutional-networks/

Code

The implementation involves creating a ConvNet using PyTorch, training the model using the training and validation data, and evaluating the model's accuracy using the test data. The code is provided in main.py and includes the following steps:

    Load the dataset using the dataset() function from data.py.
    Set the hyperparameters for the model and optimizer.
    Create the ConvNet and optimizer using the ConvNet() class and optim.SGD() function from PyTorch.
    Train the model using the model_train() function from model.py and the training and validation data.
    Evaluate the model's accuracy using the test data and print the results.
    
E2E.ipynb is self contained file for easier running .
