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
from arguments import Args
from model import Net
from dcnn import DCNN
opt = Args()

logging.basicConfig(filename='training_mcnet.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

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
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, mean=0., std=0.01)
        torch.nn.init.constant_(m.bias.data, 0.)


if __name__ == '__main__':

    train_data = dataset.Dataset_Load(opt.cover_path, opt.stego_path, opt.train_size, 'train',
                                      transform=transforms.Compose([dataset.ToPILImage(),
                                                                    dataset.RandomRotation(p=0.4),
                                                                    dataset.RandomVerticalFlip(p=0.4),
                                                                    dataset.RandomHorizontalFlip(p=0.4),
                                                                    dataset.ToTensor()]))
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    val_data = dataset.Dataset_Load(opt.valid_cover_path, opt.valid_stego_path, opt.val_size, 'valid',
                                    transform=dataset.ToTensor())
    valid_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False)

    test_data = dataset.Dataset_Load(opt.test_cover_path, opt.test_stego_path, opt.test_size, 'test',
                                     transform=dataset.ToTensor())
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    DCNN = DCNN()
    DCNN.to(device)

    DCNN.load_state_dict(torch.load(opt.dcnn_checkpoint)['model_state_dict'])

    # for p in DCNN.parameters():
    # 	p.requires_grad=False
    # 	f = DCNN.conv1.weight
    # 	# f2 = DCNN.conv12.weight

    # print(f.shape)

    model = Net()
    model.to(device)
    model = model.apply(weights_init)

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    optimizer = torch.optim.Adamax(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)

    check_point = latest_checkpoint()
    if check_point == None:
        st_epoch = 1
        print("No checkpoints found!!, Retraining started... ")
    else:
        pth = opt.checkpoints_dir + 'net_' + str(check_point) + '.pt'
        ckpt = torch.load(pth)
        st_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        print("Model Loaded from epoch " + str(st_epoch) + "..")

    for epoch in range(st_epoch, opt.num_epochs + 1):
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        test_accuracy = []

        model.train()
        adjust_learning_rate(optimizer, epoch)

        st_time = time.time()

        for i, train_batch in enumerate(train_loader):
            images = torch.cat((train_batch['cover'], train_batch['stego']), 0)
            labels = torch.cat((train_batch['label'][0], train_batch['label'][1]), 0)
            # print(images.shape)
            idx = torch.randperm(2 * opt.batch_size)
            # print(idx)
            images = images[idx]
            labels = labels[idx]
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            with torch.no_grad():
                images = DCNN(images)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss.append(loss.item())

            prediction = outputs.data.max(1)[1]
            accuracy = prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
            training_accuracy.append(accuracy.item())

            sys.stdout.write('\r Epoch:[%d/%d] Batch:[%d/%d] Loss:[%.4f] Acc:[%.2f] lr:[%.e]'
                             % (epoch, opt.num_epochs, i + 1, len(train_loader), training_loss[-1], training_accuracy[-1], optimizer.param_groups[0]['lr']))

        end_time = time.time()

        with torch.no_grad():
            model.eval()

            for i, val_batch in enumerate(valid_loader):
                images = torch.cat((val_batch['cover'], val_batch['stego']), 0)
                labels = torch.cat((val_batch['label'][0], val_batch['label'][1]), 0)

                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                # images = F.conv2d(images, f, stride=1, padding=1)
                images = DCNN(images)

                outputs = model(images)

                loss = loss_fn(outputs, labels)
                validation_loss.append(loss.item())
                prediction = outputs.data.max(1)[1]
                accuracy = prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
                validation_accuracy.append(accuracy.item())

        # with torch.no_grad():
        # 	model.eval()

            for i, test_batch in enumerate(test_loader):

                images = torch.cat((test_batch['cover'], test_batch['stego']), 0)
                labels = torch.cat((test_batch['label'][0], test_batch['label'][1]), 0)

                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                # images = F.conv2d(images, f, stride=1, padding=1)
                images = DCNN(images)

                outputs = model(images)

                # loss = loss_fn(outputs, labels)
                # test_loss.append(loss.item())

                prediction = outputs.data.max(1)[1]
                accuracy = prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
                test_accuracy.append(accuracy.item())

        avg_train_loss = sum(training_loss) / len(training_loss)
        avg_valid_loss = sum(validation_loss) / len(validation_loss)

        print('\n |Epoch: %d over| Train Loss: %.5f| Valid Loss: %.5f|\
			\n| Train Acc:%.2f| Valid Acc:%.2f| Test Acc:%.2f|time: %.2fs|\n'
              % (epoch, sum(training_loss) / len(training_loss), sum(validation_loss) / len(validation_loss),
                 sum(training_accuracy) / len(training_accuracy), sum(validation_accuracy) / len(validation_accuracy),
                 sum(test_accuracy) / len(test_accuracy), (end_time - st_time)))

        logging.info('\n |Epoch: %d over| Train Loss: %.5f| Valid Loss: %.5f|\
			\n| Train Acc:%.2f| Valid Acc:%.2f| Test Acc:%.2f|time: %.2fs|\n'
                     % (epoch, sum(training_loss) / len(training_loss), sum(validation_loss) / len(validation_loss),
                        sum(training_accuracy) / len(training_accuracy), sum(validation_accuracy) / len(validation_accuracy),
                         sum(test_accuracy) / len(test_accuracy), (end_time - st_time)))

        state = {
            'epoch': epoch,
            'opt': opt,
            'train_loss': sum(training_loss) / len(training_loss),
            'valid_loss': sum(validation_loss) / len(validation_loss),
            'train_accuracy': sum(training_accuracy) / len(training_accuracy),
            'valid_accuracy': sum(validation_accuracy) / len(validation_accuracy),
            'test_accuracy': sum(test_accuracy) / len(test_accuracy),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr']
        }
        torch.save(state, opt.checkpoints_dir + "net_" + str(epoch) + ".pt")
