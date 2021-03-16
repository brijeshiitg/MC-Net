import torch
import torch.nn as nn
import torch.nn.functional as F
from Self_Attention import Self_Attn

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.conv11 = nn.Conv2d(30, 16, kernel_size=5, stride=1, padding=2, bias=False)
		self.conv12 = nn.Conv2d(30, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv13 = nn.Conv2d(30, 16, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(48)

		self.prelu1 = nn.PReLU(num_parameters=48)
		# self.avg_pool1 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

		self.conv21 = nn.Conv2d(48, 32, kernel_size=5, stride=1, padding=2, bias=False)
		self.conv22 = nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv23 = nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(96)
		# self.avg_pool2 =nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
		self.prelu2 = nn.PReLU(num_parameters=96)

		self.conv31 = nn.Conv2d(96, 32, kernel_size=5, stride=1, padding=2, bias=False)
		self.conv32 = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv33 = nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 =	nn.BatchNorm2d(96)
		
		self.prelu3 = nn.PReLU(num_parameters=96)
		# self.avg_pool3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

		self.conv41 = nn.Conv2d(96, 64, kernel_size=5, stride=1, padding=2, bias=False)
		self.conv42 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv43 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn4 = nn.BatchNorm2d(192)
		# self.avg_pool4 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
		self.prelu4 = nn.PReLU(num_parameters=192)

		self.conv51 = nn.Conv2d(192, 128, kernel_size=5, stride=1, padding=2, bias=False)
		self.conv52 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv53 = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn5 = nn.BatchNorm2d(384)

		self.prelu5 = nn.PReLU(num_parameters=384)
		self.conv6 = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn6 = nn.BatchNorm2d(256)
		self.prelu6 = nn.PReLU(num_parameters=256)

		self.sa = Self_Attn(256, 'relu')

		self.fc = nn.Linear(256*1*1, 2)

		
	def forward(self, x):

		out = self.prelu1(self.bn1(torch.abs(torch.cat((self.conv11(x), self.conv12(x), self.conv13(x)),1))))

		out = self.prelu2(self.bn2(torch.cat((self.conv21(out), self.conv22(out), self.conv23(out)),1)))

		# temp = torch.cat((out, out),1)
		# print(out.shape)
		# print(out.shape)
		# print(temp.shape)

		out = self.prelu3(self.bn3(torch.cat((self.conv31(out), self.conv32(out), self.conv33(out)),1)))

		# temp = torch.cat((temp, out),1)

		out = self.prelu4(self.bn4(torch.cat((self.conv41(out), self.conv42(out), self.conv43(out)),1)))

		# temp = torch.cat((temp, out),1)

		out = self.prelu5(self.bn5(torch.cat((self.conv51(out), self.conv52(out), self.conv53(out)),1)))

		# temp = torch.cat((temp, out),1)

		out = self.prelu6(self.bn6(self.conv6(out)))
		out = self.sa(out)
		
		out = F.adaptive_avg_pool2d(out, (1, 1))

		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		out = F.log_softmax(out,dim=1)
		return out
