import os
import cv2
import sys
import time
import torch
import random
import logging
import argparse
import numpy as np
# from PIL import Image
from scipy import misc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import dataset as dataset
from model import Net
from sklearn import metrics
from dcnn import DCNN
import matplotlib.pyplot as plt

cover_path = '/workspace/srnet_data/srnet_test_cover/'
stego_path = '/workspace/srnet_data/srnet_test_wow05/'
checkpoint_path = './checkpoints/'
dcnn_checkpoint = '../dcnn/checkpoints/dcnn_100.pt'
checkpoint = 177	# mipod: 108, suni:168 , hill:302 , wow:278 imagenet: 513
test_size = 5000
batch_size = 10
lr=0.001
class_names = ['Normal', 'WOW_05']
class_label = {name: i for i, name in enumerate(class_names)}


def alaska_weighted_auc(y_true, y_valid):

	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
	plt.plot(metrics.roc_curve(y_true,y_valid,pos_label=1))
	plt.savefig('roc.png')
	return metrics.auc(fpr,tpr)
#,competition_metric / normalization

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		torch.nn.init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			torch.nn.init.constant_(m.bias.data, 0.0)
	elif isinstance(m, nn.Linear):
		torch.nn.init.normal_(m.weight.data, mean=0., std=0.01)
		torch.nn.init.constant_(m.bias.data, 0.)


if __name__ == '__main__':

	test_data = dataset.Dataset_Load(cover_path,stego_path,test_size,'test',
										transform=dataset.ToTensor())
	test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)
	DCNN = DCNN()
	DCNN.to(device)

	DCNN.load_state_dict(torch.load(dcnn_checkpoint)['model_state_dict'])

	model = Net()
	model.to(device)
	model = model.apply(weights_init)

	# loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adamax(model.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
	
	pth = checkpoint_path + 'net_' + str(checkpoint) +'.pt'
	ckpt = torch.load(pth)
	model.load_state_dict(ckpt['model_state_dict'])
	optimizer.load_state_dict(ckpt['optimizer_state_dict'])

	print("Model Loaded ..")
	test_accuracy=[]
	model.eval()	
	running_loss = 0
	y, preds = [],[]
	with torch.no_grad():
		for i, test_batch in enumerate(test_loader):
			# print('processing batch',i+1)
			images = torch.cat((test_batch['cover'], test_batch['stego']),0)
			labels = torch.cat((test_batch['label'][0],test_batch['label'][1]),0)
			
			images = images.to(device, dtype=torch.float)
			labels = labels.to(device, dtype=torch.long)
			images = DCNN(images)
			outputs = model(images)
			y.extend(labels.cpu().numpy().astype(int))
			preds.extend(F.softmax(outputs,1).cpu().numpy())
			print('tested batch:%d/%d'%(i+1,len(test_loader)))
		y = np.array(y)
		preds = np.array(preds)
		labels = preds.argmax(1)
		for  class_label in np.unique(y):
			idx = y==class_label
			acc = (labels[idx]==y[idx]).astype(np.float).mean()*100
			print('accuracy for class', class_names[class_label], 'is', acc)

# print('test accuracy:%.2f'%(sum(test_accuracy)/len(test_accuracy)))
		acc = (labels==y).mean()*100
		new_preds = np.zeros((len(preds),))	
		temp = preds[labels !=0, 1:]
		new_preds[labels!=0]=temp.sum(1)
		new_preds[labels==0]=1-preds[labels==0,0]
		y = np.array(y)
		y[y!=0]=1
		print(y.shape)
		print(new_preds.shape)
		auc = alaska_weighted_auc(y, new_preds)
		print('\n |AUC:%.4f| Acc: %.3f|'%(auc, acc))	

			# plt.figure(figsize=(15,7))
			# plt.plot


