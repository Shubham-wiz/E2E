# E2E
End-to-End Self-Driving via Convolutional Neural Networks

This is a machine learning lab assignment that involves implementing a Convolutional Neural Network (ConvNet) for end-to-end self-driving cars. The implementation follows the architecture proposed in the paper titled "End to End Learning for Self-Driving Cars."
Background

Before starting the assignment, there are a few background steps that need to be taken:

    Sign up on Kaggle.com and visit the page https://www.kaggle.com/asrsaiteja/car-steering-angle-prediction to see the dataset for this exercise.
    Introduce yourself to the PyTorch library, a Python library for auto-differentiation and neural network modeling. The examples page can be accessed here: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html. The Linear layers, Convolutional layers, SGD Optimizer, and Backward Gradient Calculation for Loss functions are relevant for this exercise and should be understood.
    Resources for understanding ConvNets:
    a. https://www.ismll.uni-hildesheim.de/lehre/dl-20s/script/dl-05-cnn.pdf
    b. https://cs231n.github.io/convolutional-networks/

Instructions

    Create a new notebook on the platform Kaggle.com (there should be a button to create one on the link from step 1 of the Background).
    Once the data is loaded/downloaded to this notebook, follow the code snippet provided here to read/show images.
    Divide these resulting arrays into corresponding train/validation/test splits. Leave the last 10k images for testing (images are id’ed). You are free to define the length of the validation split.
    Implement the ConvNet architecture proposed in the paper titled "End to End Learning for Self-Driving Cars." The paper can be accessed here: https://arxiv.org/abs/1604.07316
    Report one test RMSE for the test set of images.

Code

The implementation involves creating a ConvNet using PyTorch, training the model using the training and validation data, and evaluating the model's accuracy using the test data. The code is provided in main.py and includes the following steps:

    Load the dataset using the dataset() function from data.py.
    Set the hyperparameters for the model and optimizer.
    Create the ConvNet and optimizer using the ConvNet() class and optim.SGD() function from PyTorch.
    Train the model using the model_train() function from model.py and the training and validation data.
    Evaluate the model's accuracy using the test data and print the results.

How to Run

To run the code, follow these steps:

    Clone the repository to your local machine.
    Install the required packages using pip install -r requirements.txt.
    Make sure you have the dataset downloaded from Kaggle and saved to your local machine.
    Update the file paths in data.py to match the location of the dataset on your machine.
    Run main.py using the command python main.py.

Results

The code will output the training loss and validation loss for each epoch, as well as the test accuracy after training is complete. The final output will be the test RMSE for the test set of images.
