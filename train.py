import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from model import ConvNet
from data import DrivingDataset

def model_train(train, valid, test, network, optimizer, criterion, epochs, batch):
    loss_list = []
    for e in tqdm(range(epochs)):
        print(f"------------Epoch: {e} ------------")
        print("Started")
        print(f"Training for Epoch: {e}")
        # train data loading and training starts
        train_loader = DataLoader(train, batch_size=batch, shuffle=True)
        loss_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            if i % 10 == 0:
                print(f"Batch {i} of {len(train_loader)}, Loss: {loss.item()}")
        loss_train /= len(train_loader)
        print(f"Training loss for Epoch {e}: {loss_train}")
        loss_list.append(loss_train)
        
        # validation data loading and validation starts
        valid_loader = DataLoader(valid, batch_size=batch, shuffle=False)
        loss_valid = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader):
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss_valid += loss.item()
        loss_valid /= len(valid_loader)
        print(f"Validation loss for Epoch {e}: {loss_valid}")
        
    # test data loading and evaluation starts
    test_loader = DataLoader(test, batch_size=batch, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}")
    
    return loss_list

if __name__ == "__main__":
    # load dataset
    train_data = DrivingDataset()
    print(f"Train data size: {len(train_data)}")
    valid_data = DrivingDataset()
    print(f"Validation data size: {len(valid_data)}")
    test_data = DrivingDataset()
    print(f"Test data size: {len(test_data)}")
    
    # set hyperparameters
    lr = 0.01
    momentum = 0.5
    epochs = 10
    batch_size = 4
    
    # create model and optimizer
    net = ConvNet()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    # train the model
    loss_list = model_train(train_data, valid_data, test_data, net, optimizer, criterion, epochs, batch_size)
