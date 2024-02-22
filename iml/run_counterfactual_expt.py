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
import time
import tensorflow as tf
import alibi

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import euclidean
from scipy.stats import shapiro, normaltest

from torchvision import transforms
from torchvision.utils import save_image

from collections import Counter

from copy import deepcopy

from torch.autograd import Variable

from skimage.io import imread

# Local imports
from local_models import *
from helper_functions import *
from piece_hurdle_model import *
from optimize_explanations import *
from evaluation_metrics import *


# Load models and data
G, cnn = load_models(CNN, Generator)
# classifierCNN = ClassifierCNN(cnn)
# croppedCNN = CroppedCNN(cnn)
train_loader, test_loader = load_dataloaders()
X_train, y_train, X_test, y_test = get_MNIST_data(datasets)

expt1_data = pd.DataFrame(columns=['rand_num', 'name', 'MC-Mean', 'MC-STD', 'NN-Dist', 'IM1', 'IM2', 'optim_time'])
#temp_data = pd.DataFrame(columns=['rand_num', 'name', 'MC-Mean', 'MC-STD', 'NN-Dist', 'IM1', 'IM2', 'optim_time'])
df_piece = pd.DataFrame(columns=['rand_num', 'name', 'MC-Mean', 'MC-STD', 'NN-Dist', 'IM1', 'IM2', 'optim_time'])
df_ME = pd.DataFrame(columns=['rand_num', 'name', 'MC-Mean', 'MC-STD', 'NN-Dist', 'IM1', 'IM2', 'optim_time'])
df_CME = pd.DataFrame(columns=['rand_num', 'name', 'MC-Mean', 'MC-STD', 'NN-Dist', 'IM1', 'IM2', 'optim_time'])
df_CEM = pd.DataFrame(columns=['rand_num', 'name', 'MC-Mean', 'MC-STD', 'NN-Dist', 'IM1', 'IM2', 'optim_time'])
df_Proto_CF = pd.DataFrame(columns=['rand_num', 'name', 'MC-Mean', 'MC-STD', 'NN-Dist', 'IM1', 'IM2', 'optim_time'])

# k-NN for NN-Dist
X_train_act = np.load("data/distribution_data/X_train_act.npy")
X_test_act = np.load("data/distribution_data/X_test_act.npy")
X_train_pred = np.load("data/distribution_data/X_train_pred.npy")
X_test_pred = np.load("data/distribution_data/X_test_pred.npy")
k_nn = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
k_nn.fit(X_train_act, X_train_pred)


# Loading AEs for IM1 and IM2 metrics
aes, ae_full = load_autoencoders()

# Probabilitiy threshold for identifying "Exceptional Features" with PIECE
alpha = 0.05

# Iterate though 41 Incorrect examples from MNIST
for rand_num in range(1, 2):

	# Get Query representations
	original_query_idx, original_query_img, target_class = get_missclassificaiton(test_loader, cnn, rand_num)
	original_query_pred = int(torch.argmax(cnn(original_query_img)[0]))
	z = torch.load("data/latent_g_input_saved/incorrect_latent/misclassify_" + str(rand_num) + ".pt") 
	query_activations = cnn(G(z))[1][0]

	#### ========== First two steps of PIECE Algorithm ========== ####
	# Step 1: Acquire the probability of each features, and identify the excpetional ones (i.e., those with a probability lower than alpha)
	df = acquire_feature_probabilities(target_class, cnn, original_query_img, alpha) 
	# Step 2: Filter out exceptional features which we want to change, and change them to their expected values in the counterfactual class
	df = filter_df_of_exceptional_noise(df, target_class, cnn, alpha)
	# Sort by least probable to the most probable
	df = df.sort_values('Probability of Event')
	# Get x' -- The Ideal Explanation
	ideal_xp = modifying_exceptional_features(df, target_class, query_activations)   
	ideal_xp = ideal_xp.clone().detach().float().requires_grad_(False)



	for name in ['PIECE', 'Min-Edit', 'C-Min-Edit']:#,'CEM', 'Proto-CF']:

		print(" ")
		print("-------------------------------")
		print(rand_num, name)
		print("-------------------------------")

		cnn = cnn.eval()
		temp_data = pd.DataFrame(columns=['rand_num', 'name', 'MC-Mean', 'MC-STD', 'NN-Dist', 'IM1', 'IM2', 'optim_time'])

		# Query
		x_q = cnn(G(z))[1][0]

		# Explanation latent input (to optimize...)
		z_e = z.clone().detach().float().requires_grad_()

		criterion = nn.MSELoss()

		start_time = time.time()

		if name == 'PIECE':
			optimizer = optim.Adam([z_e], lr=0.01)
			z_e = optim_PIECE(G, cnn, ideal_xp, z_e, criterion, optimizer)

		elif name == 'Min-Edit':
			optimizer = optim.Adam([z_e], lr=0.001)
			z_e = optim_min_edit(cnn, G, z_e, optimizer, target_class)

		elif name == 'C-Min-Edit':
			optimizer = optim.Adam([z_e], lr=0.001)
			z_e = optim_c_min_edit(G, cnn, x_q, z_e, criterion, optimizer, target_class)

		elif name == 'CEM':
			xp = optim_CEM_Explanation(original_query_idx)
			try:
				if xp == None:
					print("Couldn't Find Explanation")
					continue
			except:
				print('Found Explanation')

		elif name == 'Proto-CF':
			xp = optim_Proto_Explanation(original_query_idx)
			try:
				if xp == None:
					print("Couldn't Find Explanation")
					continue
			except:
				print('Found Explanation')


		optim_time = time.time() - start_time

		if name == 'PIECE' or name == 'Min-Edit' or name == 'C-Min-Edit':
			I_e = G(z_e)
			
		elif name == 'CEM' or name == 'Proto-CF':
			I_e = torch.tensor(xp, dtype=torch.float32).reshape(-1,1,28,28)

		save_name = name
		save_query_and_gan_xp_for_final_data(I_e, cnn, z, G, z_e, original_query_img, save_name, rand_num,target_class, original_query_pred)
		
		# New prediction of explanation
		new_pred = int(torch.argmax(torch.exp(  cnn(I_e)[0]  )))
		
		# Metrics for Plausibility
		mc_dropout_results = mc_dropout(cnn, new_pred, I_e)
		nn_dist, _ = k_nn.kneighbors(X=np.array(    cnn(I_e)[1].detach().numpy()  )  , n_neighbors=2)
		IM1 = IM1_metric(I_e, aes, original_query_pred, new_pred)
		IM2 = IM2_metric(I_e, aes, ae_full, new_pred)
		
	
		temp_data = pd.Series({
			'rand_num': rand_num,
			'name': name,
			'MC-Mean': mc_dropout_results.mean(),
			'MC-STD': mc_dropout_results.std(),
			'NN-Dist': nn_dist[0][0],
			'IM1': IM1,
			'IM2': IM2,
			'optim_time': optim_time
		})

		# 将该 Series 添加到 DataFrame 中
		expt1_data = pd.concat([expt1_data, temp_data.to_frame().T], ignore_index=True) 
		if name == 'PIECE':
			df_piece = pd.concat([df_piece, temp_data.to_frame().T], ignore_index=True) 
		elif name == 'Min-Edit':
			df_ME = pd.concat([df_ME, temp_data.to_frame().T], ignore_index=True) 
		elif name == 'C-Min-Edit':
			df_CME = pd.concat([df_CME, temp_data.to_frame().T], ignore_index=True) 
		# elif name == 'CEM':
		# 	df_CEM = pd.concat([df_CEM, temp_data.to_frame().T], ignore_index=True)
		# elif name == 'Proto-CF':
		# 	df_Proto_CF = pd.concat([df_Proto_CF, temp_data.to_frame().T], ignore_index=True)
		print("Time to do one digit:", round(time.time() - start_time, 3))
		
	
expt1_data.to_csv('incorrect_MNIST_data.csv', index=False)

df_piece.to_csv('df_piece.csv', index=False)
df_ME.to_csv('df_ME.csv', index=False)
df_CME.to_csv('df_CME.csv', index=False)
# df_CEM.to_csv('df_ME.csv', index=False)
# df_Proto_CF.to_csv('df_CME.csv', index=False)
	
print("Time to do one digit:", round(time.time() - start_time, 3))

