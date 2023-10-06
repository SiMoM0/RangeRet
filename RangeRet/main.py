# main file
# Usage python3 main.py <config_path> <data_path>

# profiler
import pstats
import cProfile

import os
import sys
import yaml
import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from dataloader.kitti.parser import Parser

from network.rangeret import RangeRet

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input data
config_path = sys.argv[1]
dataset_folder = sys.argv[2]

# config path (label mapping)
config = load_yaml(config_path)
data_config = load_yaml(config['dataset_params']['data_config'])

# model params
model_params = config['model_params']

# get data
parser = Parser(root=dataset_folder,
                train_sequences=data_config["split"]["train"],
                valid_sequences=data_config["split"]["valid"],
                test_sequences=None,
                labels=data_config["labels"],
                color_map=data_config["color_map"],
                learning_map=data_config["learning_map"],
                learning_map_inv=data_config["learning_map_inv"],
                sensor=config["dataset_params"]["sensor"],
                max_points=config["dataset_params"]["max_points"],
                batch_size=config["train_params"]["batch_size"],
                workers=config["train_params"]["workers"],
                gt=True,
                shuffle_train=True)

model = RangeRet(model_params).to(device)

# TODO use focal loss, lovasz loss
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
#loss_fn = nn.NLLLoss(ignore_index=0)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)

def train_one_epoch(train_loader, epoch_index):
    running_loss = 0.

    # confusion matrix
    conf_matrix = np.zeros((20, 20), dtype=np.int64)

    #for i, data in enumerate(zip(range_images, labels_images)):
    for i, (in_vol, _, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):

        optimizer.zero_grad()

        in_vol = in_vol.cuda()
        proj_labels = proj_labels.cuda()

        #print(in_vol) # (B, H, W, C)

        outputs = model(in_vol) # input format (B, H, W, C)

        #print('outputs shape: ', outputs.shape)
        #print('labels shape: ', labels_images[i].shape)

        #print(proj_labels.shape) # (B, H, W)

        predictions = outputs.permute(0, 3, 1, 2)
        #gt = proj_labels.permute(2, 0, 1)

        proj_argmax = predictions[0].argmax(dim=0)
        #print(proj_argmax)
        #print(proj_labels)

        #np.savetxt('pred.txt', proj_argmax[0].cpu().detach().numpy(), fmt='%d')
        #np.savetxt('labels.txt', proj_labels[0].cpu().detach().numpy(), fmt='%d')

        #print('predictions shape ', predictions.shape)
        #print('labels shape ', gt.shape)

        #loss = loss_fn(torch.log(predictions.clamp(min=1e-8)), gt.cuda(non_blocking=True))
        loss = loss_fn(predictions, proj_labels.cuda(non_blocking=True).long())
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        # populate confusion matrix
        idxs = tuple(np.stack((proj_argmax.reshape(-1, 1).cpu().detach().numpy(), proj_labels.reshape(-1, 1).cpu().detach().numpy()), axis=0))
        np.add.at(conf_matrix, idxs, 1)

        # TODO put in original pointcloud using indexes and compute loss between whole point cloud labels 
        #unproj_argmax = proj_argmax[p_y, p_x]
        #print(unproj_argmax.shape)

        #pred_np = unproj_argmax.cpu().detach().numpy()
        #pred_np = pred_np.reshape((-1)).astype(np.int32)

    # print final predictions
    # np.savetxt(f'pred{epoch_index}.txt', torch.argmax(predictions, dim=1).cpu().detach().numpy()[0], fmt="%d")

    # clean stats
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=1) - tp
    fn = conf_matrix.sum(axis=0) - tp

    intersection = tp
    union = tp + fp + fn + 1e-15

    iou = intersection / union
    iou_mean = (intersection / union).mean()

    return running_loss, iou_mean

def validate(val_loader):
    model.eval()

    val_loss = 0.

    # confusion matrix
    conf_matrix = np.zeros((20, 20), dtype=np.int64)

    #for i, data in enumerate(zip(range_images, labels_images)):
    for i, (in_vol, _, proj_labels, _, path_seq, path_name, p_x, p_y, _, _, _, _, _, _, _) in tqdm(enumerate(val_loader), total=len(val_loader)):

        in_vol = in_vol.cuda()
        proj_labels = proj_labels.cuda()
        p_x = p_x.cuda()
        p_y = p_y.cuda()

        #print(in_vol) # (B, H, W, C)

        outputs = model(in_vol) # input format (B, H, W, C)

        #print('outputs shape: ', outputs.shape)
        #print('labels shape: ', labels_images[i].shape)

        #print(proj_labels.shape) # (B, H, W)

        predictions = outputs.permute(0, 3, 1, 2)
        #gt = proj_labels.permute(2, 0, 1)

        proj_argmax = predictions[0].argmax(dim=0)
        #print(proj_argmax)
        #print(proj_labels)

        #np.savetxt('pred.txt', proj_argmax[0].cpu().detach().numpy(), fmt='%d')
        #np.savetxt('labels.txt', proj_labels[0].cpu().detach().numpy(), fmt='%d')

        #print('predictions shape ', predictions.shape)
        #print('labels shape ', gt.shape)

        #loss = loss_fn(torch.log(predictions.clamp(min=1e-8)), gt.cuda(non_blocking=True))
        loss = loss_fn(predictions, proj_labels.cuda(non_blocking=True).long())
        val_loss += loss.item()

        # populate confusion matrix
        idxs = tuple(np.stack((proj_argmax.reshape(-1, 1).cpu().detach().numpy(), proj_labels.reshape(-1, 1).cpu().detach().numpy()), axis=0))
        np.add.at(conf_matrix, idxs, 1)

        # put in original pointcloud using indexes
        unproj_argmax = proj_argmax[p_y, p_x]
        #print(unproj_argmax.shape)

        pred_np = unproj_argmax.cpu().detach().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        #np.savetxt('pc_predicitons.txt', pred_np)

    # clean stats
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=1) - tp
    fn = conf_matrix.sum(axis=0) - tp

    intersection = tp
    union = tp + fp + fn + 1e-15

    iou = intersection / union
    iou_mean = (intersection / union).mean()

    return val_loss, iou_mean

epoch_number = 0

EPOCHS = config['train_params']['num_epochs']

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)

    #cProfile.run("train_one_epoch(train_loader=parser.get_train_set(), epoch_index=epoch_number)", "my_func_stats")
    #p = pstats.Stats("my_func_stats")
    #p.sort_stats("cumulative").print_stats()
    
    avg_loss, iou_mean = train_one_epoch(train_loader=parser.get_train_set(), epoch_index=epoch_number)

    print('TRAIN Loss = {} | mIoU = {:.2%}'.format(avg_loss, iou_mean))

    val_loss, val_iou = validate(val_loader=parser.get_valid_set())

    print('VALIDATION Loss = {} | mIoU = {:.2%}'.format(val_loss, val_iou))

    epoch_number += 1

# save model
torch.save(model.state_dict(), 'rangeret-model.pt')