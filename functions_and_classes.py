import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from collections import OrderedDict

#Data transforms
data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomRotation(0,360),
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ]),
                   'valid': transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
                  }

#Process_Image function
def process_image(image):
    '''takes in a PIL Image & transforms it to allow a forward pass'''
    image = image.resize((256,256))
    image = image.crop((0,0,224,224))
    np_image = np.array(image)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pil_image = (np_image - mean) / std
    return pil_image

