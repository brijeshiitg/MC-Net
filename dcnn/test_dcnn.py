import cv2
import torch
from scipy import misc, io
import numpy as np

from dcnn import DCNN

image_path = '/workspace/srnet_data/training/cover/'
# image_path = '/media/multimedia/Backup/Datasets/srnet_test_cover/'
result_path= './'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

with torch.no_grad():
	model = DCNN()
	chkpt = torch.load('./dcnn_checkpoints/dcnn_300.pt')
	# print(chkpt)
	model.load_state_dict(chkpt['model_state_dict'])
	model.to(device)
	model.eval()
	# print(misc.imread(image_path+'5102.pgm').astype(np.float32))
	img = misc.imread(image_path+'5102.pgm').astype(np.float32).reshape((1,1,256,256))
	img = torch.from_numpy(img).to(device)
	output = model(img)
	output = output.squeeze(0)
	output = output.cpu().numpy().reshape((256,256))
	# print(output.shape)
	# misc.imsave('result_5102.pgm', np.uint8(output))
	cv2.imwrite(result_path+'result_5102.pgm', np.uint8(output))
	# io.savemat(result_path+'wow_result_5102.mat', {'im':output})
	print(output)

