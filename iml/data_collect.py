import os
import torch
import torchvision
import torch.nn as nn
import pickle
import pylab
import numpy as np
import scipy
import torch.optim as optim
import pandas as pd
import torchvision.datasets as datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import shapiro, normaltest

from torchvision import transforms

from copy import deepcopy

# Local imports
from local_models import *
from helper_functions import *
from piece_hurdle_model import *
from optimize_explanations import *
from evaluation_metrics import *

from IPython.display import Image

G, cnn = load_models(CNN, Generator)
# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download = True)
# mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download = True)
train_loader, test_loader = load_dataloaders()
X_train, y_train, X_test, y_test = get_MNIST_data(datasets)


def return_feature_contribution_data(data_loader, cnn, num_classes=10):
    
    full_data = dict()
    pred_idx = dict() 

    for class_name in list(range(num_classes)):
        pred_idx[class_name] = list()
        
    for i, data in enumerate(data_loader):
        # print progress
        if i % 10000 == 0:
            print(  100 * round(i / len(data_loader), 2), "% complete..."  )     
        image, label = data
        label = int(label.detach().numpy())
        acts = cnn(image)[1][0].detach().numpy()
        pred = int(torch.argmax(  cnn(image)[0]  ).detach().numpy()) 
        pred_idx[pred].append(acts.tolist())
                
    return pred_idx
# 假设你已经收集到了数据，命名为 collected_data
# collected_data 应该是一个字典，每个键代表一个类别，对应的值是一个包含该类别激活的列表

collected_data = return_feature_contribution_data(train_loader, cnn)
dist_data = {}

# 假设 num_classes 是类别的数量
num_classes = 10

# 为每个类别创建一个空列表
for class_name in range(num_classes):
    dist_data[class_name] = {'activations': []}

# 将 pred_idx_train 中的数据填充到 dist_data 中
for class_name, activations_list in collected_data.items():
    # 将 activations_list 转换为 numpy 数组
    activations_array = np.array(activations_list)
    # 将 activations_array 存储到 dist_data 对应的类别中
    dist_data[class_name]['activations'] = activations_array


with open('collected_data.pickle', 'wb') as handle:
    pickle.dump(dist_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("数据已成功存储为 pickle 文件。")