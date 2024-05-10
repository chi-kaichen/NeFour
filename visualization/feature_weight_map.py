# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:25:45 2021

@author: 13710
"""
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from model import UNet_D1

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

def draw_cam(model,our, img_path, save_path, transform=None, visheadmap=False):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    img = img.unsqueeze(0)
    model.eval()

    x = our.conv1(img)
    x = our.conv2(x)
    x = our.channel_attention1(x)
    x = our.channel_attention2(x)
    x = our.pixel_attention2(x)
    x = our.pixel_attention3(x)
    #x = our.pixel_attention4(x)
    #x = our.DC(x)

   
    
    
    features = x                #1x16x224x224
    #print(features.shape)
    
    output = model.avgpool(x)   #1x16x1x1
    #print(output.shape)
    
    output = output.view(output.size(0), -1)
    #print(output.shape)         #1x16

    
    def extract(g):
        global feature_grad
        feature_grad = g
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    features.register_hook(extract)
    pred_class.backward()
    greds = feature_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(16):
        features[i, ...] *= pooled_grads[i, ...]
    headmap = features.detach().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)
    
    if visheadmap:
        plt.matshow(headmap)
        # plt.savefig(headmap, './headmap.png')
        plt.show()
 
    img = cv2.imread(img_path)
    headmap = cv2.resize(headmap, (img.shape[1], img.shape[0]))
    headmap = np.uint8(255*headmap)
    headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
    superimposed_img = headmap*0.4 + img
    cv2.imwrite(save_path, superimposed_img)
    
if __name__ == '__main__':
     model = models.resnet50(pretrained=True)
     our = UNet_D1()
     transform = transforms.Compose([transforms.Resize((1792, 1792)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
     draw_cam(model, our,'./854fine.png', './a.png', transform=transform, visheadmap=True)