import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# class DCNN(nn.Module):
# 	def __init__(self):
# 		super(DCNN,0 self).__init__()

# 		self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)
# 		self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)

# 	def forward(self, x):

# 		# out1 = self.conv1(x)
# 		out1 = F.conv2d()
# 		out = self.conv2(out1)
# 		return out
srm3=np.load('srm_spam_12_filters.npy').reshape((12,1,3,3))
srm5=np.load('srm_kernels_18x5x5.npy').reshape((18,1,5,5))

# srm = np.load('./SRM_Kernels.npy')
weights3 = torch.from_numpy(srm3).float().cuda()
weights5 = torch.from_numpy(srm5).float().cuda()
# print(type(weights))
class DCNN(nn.Module):
	def __init__(self):
		super(DCNN, self).__init__()

		# self.conv1 = nn.Conv2d(30, 30, kernel_size=5, stride=1, padding=2, bias=True)
		self.conv2 = nn.Conv2d(30, 1, kernel_size=5, stride=1, padding=2, bias=True)

	def forward(self, x):
		# print(x.shape)
		out11 = F.conv2d(x, weights3, padding=1)
		out12 = F.conv2d(x, weights5, padding=2)

		# print(out1.shape)
		temp = torch.cat((out11, out12),1)
		out = self.conv2(temp)
		# print(out.shape)
		return out
