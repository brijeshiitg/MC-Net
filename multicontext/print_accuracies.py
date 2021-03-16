# plot losses
import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
checkpoint_dir = './checkpoints/'

epochs =355
TRAIN_ACCURACY=[]
VALID_ACCURACY=[]
TEST_ACCURACY=[]

for e in range(1,epochs+1):
	checkpoint = torch.load(os.path.join(
							checkpoint_dir, 'net_'+str(e)+'.pt'))
	TRAIN_ACCURACY.append(checkpoint['train_accuracy'])#(sum(checkpoint['train_loss'])/len(checkpoint['train_loss']))
	VALID_ACCURACY.append(checkpoint['valid_accuracy'])#(sum(checkpoint['valid_loss'])/len(checkpoint['valid_loss']))
	TEST_ACCURACY.append(checkpoint['test_accuracy'])#(sum(checkpoint['valid_loss'])/len(checkpoint['valid_loss']))

EPOCHS = [i for i in range(1,epochs+1)]
ind = VALID_ACCURACY.index(max(VALID_ACCURACY))
print('epoch:{} val_acc:{} test_acc:{}'.format(EPOCHS[ind],VALID_ACCURACY[ind],TEST_ACCURACY[ind]))
print('independent test acc:{}'.format(max(TEST_ACCURACY)))
