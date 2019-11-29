#!/usr/bin/env python


import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models,datasets
from torch.optim import lr_scheduler 
from PIL import Image
from torch.autograd import Variable
import time
import os
import copy
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--img_path', action='store', dest='img_path', help='path of image to predict', required=False)
parser.add_argument('--mod_path', action='store', dest='mod_path', help='path saved model', required=False)
results = parser.parse_args()


img_path = results.img_path
mod_path = results.mod_path
classes  = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial'
           , 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# loading model
def loading_model(path):
    
    # returning the pretrained resnet18 model
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    model_ft.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model_ft


# preparing image for model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    
    image_transforms = trans.Compose([
        trans.Resize(256),
        trans.CenterCrop(224),
        trans.ToTensor(),
        trans.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    img = image_transforms(pil_image)
    return img


# Class Prediction
def predict(processed_image, model, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    model.eval()
    model.to(device) 
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    processed_image = processed_image.to(device)

    
    with torch.no_grad():
        output = model.forward(processed_image)
        _, predicted = torch.max(output.data, 1)
 
        
    return classes[predicted]



''' Calling our functions to classify the image '''


model = loading_model(mod_path)
image = process_image(img_path)
result = predict(image, model, device)
print(result)