import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
from scipy import misc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import dcnn_dataset as dataset
from dcnn_arguments import Args
from dcnn import DCNN
opt = Args()

logging.basicConfig(filename='dcnn_training.log',format='%(asctime)s %(message)s', level=logging.DEBUG)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def latest_checkpoint():
	latest = None
	if os.path.exists(opt.checkpoints_dir):
		file_names = os.listdir(opt.checkpoints_dir)
		if len(file_names) < 1:
			latest = None
		else:
			vals = []
			for file in file_names:
				tmp = file.split('.')
				vals.append(int(tmp[0].split('_')[1]))
			latest = max(vals)
	return latest

def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 1/5th every 30 epochs"""
	lr = opt.lr * (0.5 ** (epoch // 20))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		torch.nn.init.normal_(m.weight.data, mean=0., std=0.01)
		if m.bias is not None:
			torch.nn.init.constant_(m.bias.data, 0.0)
	elif isinstance(m, nn.Linear):
		torch.nn.init.normal_(m.weight.data, mean=0., std=0.01)
		torch.nn.init.constant_(m.bias.data, 0.)




if __name__ == '__main__':

	train_data = dataset.Dataset_Load(opt.cover_path,opt.stego_path,  opt.train_size,'train',
							 transform= transforms.Compose([dataset.ToPILImage(),
							 	dataset.RandomRotation(p=opt.p_rot),
							 	dataset.RandomVerticalFlip(p=opt.p_vflip),
							 	dataset.RandomHorizontalFlip(p=opt.p_hflip),
							 	dataset.ToTensor()]))
	train_loader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True)

	val_data = dataset.Dataset_Load(opt.valid_cover_path,opt.valid_stego_path,opt.val_size,'valid',
										transform=dataset.ToTensor())
	valid_loader = DataLoader(val_data,batch_size=opt.batch_size,shuffle=False)

	model = DCNN()
	print(model)
	# model = model.apply(weights_init)
	model.to(device)

	loss_fn = nn.MSELoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
	optimizer = torch.optim.Adamax(model.parameters(), lr = opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
	
	if not os.path.exists(opt.checkpoints_dir):
		os.makedirs(opt.checkpoints_dir)
	
	check_point = latest_checkpoint()
	if check_point == None:
		st_epoch = 1
		print("No checkpoints found!!, Retraining started... ")
	else:
		pth = opt.checkpoints_dir + 'dcnn_' + str(check_point) +'.pt'
		ckpt = torch.load(pth)
		st_epoch = ckpt['epoch'] + 1
		model.load_state_dict(ckpt['model_state_dict'])
		optimizer.load_state_dict(ckpt['optimizer_state_dict'])

		print("Model Loaded from epoch " + str(st_epoch) + "..")

	for epoch in range(st_epoch, opt.num_epochs+1):
		training_loss = []
		training_accuracy=[]
		validation_loss=[]
		validation_accuracy=[]

		model.train()
		adjust_learning_rate(optimizer, epoch)
		
		st_time = time.time()

		for i, train_batch in enumerate(train_loader):
			images = torch.cat((train_batch['cover'], train_batch['stego']),0)
			labels = torch.cat((train_batch['cover'], train_batch['cover']),0)

			images = images.to(device, dtype=torch.float)
			labels = labels.to(device, dtype=torch.float)

			labels = torch.abs(images-labels)

			optimizer.zero_grad()

			outputs = model(images)

			loss = loss_fn(outputs, labels)

			loss.backward()

			optimizer.step()

			training_loss.append(loss.item())

			sys.stdout.write('\r Epoch:[%d/%d] Batch:[%d/%d] Loss:[%.4f] lr:[%.e]'
				%(epoch, opt.num_epochs, i+1, len(train_loader), training_loss[-1], optimizer.param_groups[0]['lr']))
		
		end_time = time.time()

		with torch.no_grad():
			model.eval()

			for i, val_batch in enumerate(valid_loader):
				images = torch.cat((val_batch['cover'], val_batch['stego']),0)
				labels = torch.cat((val_batch['cover'], val_batch['cover']),0)

				images = images.to(device, dtype=torch.float)
				labels = labels.to(device, dtype=torch.float)

				labels = torch.abs(images-labels)

				outputs = model(images)

				loss = loss_fn(outputs, labels)
				validation_loss.append(loss.item())

			
		avg_train_loss = sum(training_loss)/len(training_loss)
		avg_valid_loss = sum(validation_loss)/len(validation_loss)

		print('\n |Epoch: %d over| Train Loss: %.5f| Valid Loss: %.5f| time: %.2fs|\n'
			%(epoch, sum(training_loss)/len(training_loss), sum(validation_loss)/len(validation_loss), (end_time-st_time)))
		
		logging.info('\n |Epoch: %d over| Train Loss: %.5f| Valid Loss: %.5f| time: %.2fs|\n'
			%(epoch, sum(training_loss)/len(training_loss), sum(validation_loss)/len(validation_loss), (end_time-st_time)))

		state = {
				'epoch':epoch,
				'opt': opt,
				'train_loss': sum(training_loss)/len(training_loss),
				'valid_loss': sum(validation_loss)/len(validation_loss),
				'model_state_dict':model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'lr':optimizer.param_groups[0]['lr']
				}
		torch.save(state,opt.checkpoints_dir+"dcnn_"+str(epoch)+".pt")




