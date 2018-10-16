# Imports here
import json
import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
#Import Functions
import functions_and_classes
from functions_and_classes import process_image

#Command Line Inputs - Positional Data Directory (Mandatory)
parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
parser.add_argument('input', help='Path to the image for prediction')
parser.add_argument('checkpoint', help='Path to the checkpoint to load')
parser.add_argument('-t', '--top_k', metavar='', type=int, help='Set the number of most-likely classes to return when predicting [-t]')
parser.add_argument('-c', '--category_names', metavar='', help='Path to json file [-c]')
parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for inference [-g]')
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help='print verbose')
args = parser.parse_args()

#top_k Handler Defaults to 5
top_k = (args.top_k or 5)

#Category to name user argument defaults to 'cat_to_name.json'
json_file = (args.category_names or 'cat_to_name.json')
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)
    
# Function to load a model from a checkpoint file
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer_state = checkpoint['optimizer_state']
    model.load_state_dict(checkpoint['state_dict'])
    return model
    
#Prediction Function
def predict(image_path, model, topk=top_k): 
    '''returns the top kk probabilities & classes for given image based on user input & model architecture'''
    image_tensor = torch.from_numpy(image_path)
    image_tensor = np.transpose(image_tensor, (2,0,1))
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model.double().forward(image_tensor)
    #receive probabilities as 0 - 1 floats with exp
    ps = torch.exp(output)
    topk_combined = ps.topk(top_k, sorted=True)
    #top kk probabilities
    topk_ps = topk_combined[0][0]
    #top kk classes
    topk_cs = topk_combined[1][0]
    return topk_ps, topk_cs  

    
#Main Function
if __name__ == '__main__':
    
    #GPU Handler
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('GPU Enabled')
    else:
        device = torch.device("cpu")
        print('Please note, GPU usage not requested or not available [-g] : Using CPU instead')
    
    #Loading trained model for prediction
    model = load_checkpoint(args.checkpoint)

    #Define PIL image & run it with process_image() to get predictions
    image = Image.open(args.input)
    processed_img = process_image(image)    
    
    #Creating a Dataframe that contains Labels & Indices of Classes (df_combined)
    pd_class_to_idx = pd.Series(model.class_to_idx)
    pd_cat_to_name = pd.Series(cat_to_name)

    #choosing column indices
    columns = {'Index' : pd_class_to_idx,
               'Label': pd_cat_to_name}
    #combining series to one dataframe
    df_combined = pd.DataFrame(columns)
    
    #Predicting top kk with predict()
    topk_ps, topk_cs = predict(processed_img, model)
   
    #itterating trough the top kk classes & translating it to legible strings from df_combined
    comb_classes = []
    
    for i in topk_cs:
        comb_class = df_combined.values[i]
        comb_classes.append(comb_class)
    
    flower_name = comb_classes[0][1]
    class_probability = np.linalg.norm(topk_ps[0])
    
    #print probabilities
    if args.quiet:
        print('flower name:', flower_name, ', class probability: {:.2f}'.format(class_probability))   
    elif args.verbose:
        print('Returning', len(topk_cs), 'most likely classes...')
        for i in range(len(topk_cs)):
            print('probability #', i+1, ', flower:', comb_classes[i][1], ', prob: {:.2f}'.format(np.linalg.norm(topk_ps[i])))
    else:
        print('flower name:', flower_name, ', class probability: {:.2f}'.format(class_probability)) 
        print('re-run with -v in the terminal to get your top-k classes & probabilities')
    