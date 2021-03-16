import  argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cover_path', default='/workspace/srnet_data/srnet_train_cover/')
	parser.add_argument('--stego_path', default='/workspace/srnet_data/srnet_train_wow05/')
	parser.add_argument('--valid_cover_path', default='/workspace/srnet_data/srnet_val_cover/')
	parser.add_argument('--valid_stego_path', default='/workspace/srnet_data/srnet_val_wow05/')
	parser.add_argument('--test_cover_path', default='/workspace/srnet_data/srnet_test_cover/')
	parser.add_argument('--test_stego_path', default='/workspace/srnet_data/srnet_test_wow05/')
	parser.add_argument('--checkpoints_dir', default='./checkpoints/')
	parser.add_argument('--dcnn_checkpoint', default='../dcnn/checkpoints/dcnn_100.pt')
	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--image_size', type=int, default=256)
	parser.add_argument('--num_epochs', type=int, default=1000)
	parser.add_argument('--train_size', type=int, default=11000)
	parser.add_argument('--val_size', type=int, default=1000)
	parser.add_argument('--test_size', type=int, default=5000)
	parser.add_argument('--p_rot', type=float, default=0.4)
	parser.add_argument('--p_hflip', type=float, default=0.4)
	parser.add_argument('--p_vflip', type=float, default=0.4)
	parser.add_argument('--lr', type=float, default=0.001)

	opt = parser.parse_args()
	return opt
