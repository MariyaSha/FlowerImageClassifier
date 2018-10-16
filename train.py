#Imports Here
import argparse
import os
import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
#Import Transforms
import functions_and_classes
from functions_and_classes import data_transforms

#Command Line Inputs - Positional Data Directory (Mandatory)
parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
parser.add_argument('data_dir', metavar='', help='Path to the data directory for training')
#Optional Arguments (directory to save checkpoints, model architecture, lr, hu, epochs, gpu enabling)
parser.add_argument('-s', '--save_dir', metavar='', help='Save the checkpoint in the given directory')
parser.add_argument('-a', '--arch', metavar='', help="Set model's architecture")
parser.add_argument('-l', '--learning_rate', metavar='', type=float, help='Set learning rate')
parser.add_argument('-H', '--hidden_units', metavar='', type=int, help='Set the number of hidden units')
parser.add_argument('-e', '--epochs', metavar='', type=int, help="Set the number of epochs")
parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for training')
args = parser.parse_args()

#Directory paths
test_dir = os.path.join(args.data_dir, "test") 
train_dir = os.path.join(args.data_dir, "train") 
valid_dir = os.path.join(args.data_dir, "valid") 

#Set Directory to Save Checkpoints
def save_dir(dir):
    ''' if directory exists, save all the checkpoints there.
        if directry does not exist, create it to save all the checkpoints there
    '''
    if os.path.exists(dir):
        save_directory = dir
        print('Please note, directory:', dir.strip('./'),'already exists in the given path')
    elif not os.path.exists(dir): 
        print('Creating new directory:', dir, 'in the given path')
        save_directory = os.mkdir(dir)
    return save_directory

#Validation Function
def validation(model, testloader, criterion):
    #Define validation variables
    validation_loss = 0
    validation_accuracy = 0
    #Itterare over the validation set
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        validation_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        validation_accuracy += equality.type(torch.FloatTensor).mean()  
    return validation_loss, validation_accuracy
        
#Our main function
if __name__ == '__main__':
    #User Argument Handlers   
    #Save Directory Handler
    if args.save_dir == None:
        print('No argument was given for Saving Directory [-s] : Using default "./save_directory"')
        save_directory = save_dir('./save_directory')
    else:
        save_directory = save_dir(args.save_dir)
          
    #Model Architecture Handler & Accomodating Input Units Number
    if args.arch == None:
        print('No argument was given for Model Architecture [-a] : Using default "vgg13"')
        model = models.vgg13(pretrained=True)
        input_units = 25088
    elif args.arch.startswith('vgg'):
        model = models.__dict__[args.arch](pretrained=True)
        input_units = 25088
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](pretrained=True)
        input_units = 1024
    else:
        print('Sorry, the network runs only on vgg or densenet models : Using default "vgg13"')
        model = models.vgg13(pretrained=True)
        input_units = 25088        
 
    #Device ID Handler (Using GPU for Training)
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('GPU Enabled')
    else:
        device = torch.device("cpu")
        print('GPU usage not requested. Call [-g] to use GPU : Using CPU')
        
    #Hyperparameters Handler
    learning_rate = (args.learning_rate or 0.001)
    hidden_units = (args.hidden_units or 512)
    epochs = (args.epochs or 4)  
    
    # Freeze parameters for backpropogation
    for param in model.parameters():
        param.requires_grad = False

    #Set hidden units user input inside classifier function (if exists)
    hidden_units = (args.hidden_units or 512)
    
    #Re-write the classifier and define it as my classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_units, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier   

    # Loading datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}

    # Define Dataloaders
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64,shuffle=True),
                  'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
                  'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)}   
    
    #Training Begins
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #Define itteration variables
    print_every = 40
    steps = 0

    model.to(device)

    #Start training the model
    for e in range(epochs):
        model.train()
        running_loss = 0
        #Itterare over the training set
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)        
            optimizer.zero_grad()      
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            #Model Progress
            if steps % print_every == 0:
                #Model in evaluation
                model.eval()            
                # Turn off gradients for validation
                with torch.no_grad():
                    validation_loss, validation_accuracy = validation(model, dataloaders['valid'], criterion)
                #Printing Training + Validation information    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(dataloaders['valid'])),
                      "Validation Accuracy: {:.3f}".format(validation_accuracy/len(dataloaders['valid'])))          
                running_loss = 0
            
            # Model back to training
            model.train()

    #Validation on the test set
    correct = 0
    total = 0
    #Itterare over the test set
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    #Save Checkpoint
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'class_to_idx': image_datasets['train'].class_to_idx,
                  'epochs': 4,
                  'print_every': 40,
                  'optimizer_state': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(save_directory,'checkpoint.pth'))
    

           