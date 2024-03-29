import os
import torch
import random
import numbers
import numpy as np
import torch.nn as nn
from PIL import Image
from scipy import misc
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from dcnn_arguments import Args

opt = Args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RandomRotation(object):

	def __init__(self, resample=False, expand=False, center=None, p=opt.p_rot):
		degrees = random.randint(1, 359)
		if isinstance(degrees, numbers.Number):
			if degrees < 0:
				raise ValueError("If degrees is a single number, it must be positive.")
			self.degrees = (-degrees, degrees)
		else:
			if len(degrees) != 2:
				raise ValueError("If degrees is a sequence, it must be of len 2.")
			self.degrees = degrees
		self.resample = resample
		self.expand = expand
		self.center = center
		self.p = p


	@staticmethod
	def get_params(degrees):
		angle = random.uniform(degrees[0], degrees[1])
		return angle

	def __call__(self, sample):
		cover_image, stego_image = sample['cover'], sample['stego']
		angle = self.get_params(self.degrees)

		if random.random() < self.p:
			return {'cover': F.rotate(cover_image, angle, self.resample, self.expand, self.center ),
					'stego': F.rotate(stego_image, angle, self.resample, self.expand, self.center )}

		return{'cover':cover_image,
				'stego':stego_image}

	def __repr__(self):
		format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
		format_string += ', resample={0}'.format(self.resample)
		format_string += ', expand={0}'.format(self.expand)
		
		if self.center is not None:
			format_string += ', center={0}'.format(self.center)
		format_string += ')'
		
		return format_string

class ToPILImage(object):

	def __call__(self, sample):

		cover_image, stego_image = sample['cover'], sample['stego']
		
		return {'cover':F.to_pil_image(cover_image),
				'stego':F.to_pil_image(stego_image)}

class ToTensor(object):

	def __call__(self, sample):
		
		cover_image, stego_image = sample['cover'], sample['stego']
				
		cover_image = np.asarray(cover_image).reshape((opt.image_size,opt.image_size,1)).astype(np.float32)
		cover_image = torch.from_numpy(cover_image)
		cover_image = torch.transpose(torch.transpose(cover_image, 2, 0), 1, 2)
		cover_image = cover_image / 255.0

		stego_image = np.asarray(stego_image).reshape((opt.image_size,opt.image_size,1)).astype(np.float32)
		stego_image = torch.from_numpy(stego_image)
		stego_image = torch.transpose(torch.transpose(stego_image, 2, 0), 1, 2)
		stego_image = stego_image / 255.0

		return {'cover': cover_image,
				'stego': stego_image }

class RandomHorizontalFlip(object):

	def __init__(self, p=opt.p_hflip):
		
		self.p = p

	def __call__(self, sample):

		cover_image, stego_image = sample['cover'], sample['stego']
		if random.random() < self.p:
			return {'cover':F.hflip(cover_image),
					'stego':F.hflip(stego_image)}

		return {'cover':cover_image,
				'stego':stego_image}

	def __repr__(self):
		
		return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):

	def __init__(self, p=opt.p_vflip):
		
		self.p = p

	def __call__(self, sample):

		cover_image, stego_image = sample['cover'], sample['stego']
		if random.random() < self.p:
			return {'cover':F.vflip(cover_image),
					'stego':F.vflip(stego_image)}

		return {'cover':cover_image,
				'stego':stego_image}

	def __repr__(self):
		
		return self.__class__.__name__ + '(p={})'.format(self.p)


class Dataset_Load(Dataset):
	def __init__(self,cover_path, stego_path,size,flag, transform=None):
		self.flag = flag
		self.cover = cover_path
		self.stego = stego_path
		self.transforms = transform
		self.data_size = size

	def __len__(self):
		if self.flag=='train':
			self.data_size = opt.train_size
		elif self.flag=='val':
			self.data_size = opt.val_size
		return self.data_size

	def __getitem__(self, index):

		if self.flag=='train':
			if index<1400:
				index += 1
			else:index+=4001
		elif self.flag=='valid':
			index+=1401

		img_name = str(index) + ".pgm"
		cover_img = misc.imread(os.path.join(self.cover,img_name))
		stego_img = misc.imread(os.path.join(self.stego,img_name))

		label1 = torch.tensor(0,dtype=torch.long).to(device)
		label2 = torch.tensor(1,dtype=torch.long).to(device)
		sample = {'cover':cover_img, 'stego':stego_img}

		if self.transforms !=None:
			sample = self.transforms(sample)

		# sample['label'] = [label1, label2]
		return sample







