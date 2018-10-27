# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:38:49 2018

@author: asus
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing as ppc

def load_data(file_name):
    data=[]
    with open(file_name) as txt_data:
        lines=txt_data.readlines()
        for line in lines:
            #.strip()默认去除行首尾空格
            line=line.strip().split(',') 
            data.append(line)
        return np.array(data)
    
def split_data(dataset):
    feature=[]
    label=[]
    for i in range(len(dataset)):
        feature.append([data for data in dataset[i][:-1]])
        label.append(dataset[i][-1])
    return np.array(feature),np.array(label)

path = 'data.txt'
data = load_data(path)
feature,label=data[:,:-1],data[:,-1]

label_encoder = ppc.LabelEncoder()
label_encoder.fit(['x','o','b'])

for i in range(len(feature)):
    feature[i]=label_encoder.transform(feature[i])
    
for i in range(len(label)):
    label[i]=1 if label[i]=='positive' else 0

print(label[:5])
