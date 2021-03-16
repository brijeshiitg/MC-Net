import  argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cover_path', default='/workspace/srnet_data/srnet_train_cover/')
	parser.add_argument('--stego_path', default='/workspace/srnet_data/srnet_train_wow05/')
	parser.add_argument('--valid_cover_path', default='/workspace/srnet_data/srnet_train_cover/')
	parser.add_argument('--valid_stego_path', default='/workspace/srnet_data/srnet_train_wow05/')
	parser.add_argument('--checkpoints_dir', default='./new_dcnn_checkpoints/')
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--image_size', type=int, default=256)
	parser.add_argument('--num_epochs', type=int, default=400)
	parser.add_argument('--train_size', type=int, default=2700)
	parser.add_argument('--val_size', type=int, default=300)
	parser.add_argument('--p_rot', type=float, default=0.5)
	parser.add_argument('--p_hflip', type=float, default=0.5)
	parser.add_argument('--p_vflip', type=float, default=0.5)
	parser.add_argument('--lr', type=float, default=0.01)

	opt = parser.parse_args()
	return opt
