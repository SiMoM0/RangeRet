# main file
# Usage python3 infer.py <config_path> <model_path> <data_path> <pred_path> <split>
# split can be one among train, valid, test

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

from utils.knn import KNN
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
model_path = sys.argv[2]
dataset_folder = sys.argv[3]
prediction_path = sys.argv[4]
split = sys.argv[5]

# config path (label mapping)
config = load_yaml(config_path)
data_config = load_yaml(config['dataset_params']['data_config'])

# create predictions folder
prediction_path = os.path.join(prediction_path, 'sequences')
if not os.path.exists(prediction_path):
    os.mkdir(prediction_path)

# model params
model_params = config['model_params']

# get data
parser = Parser(root=dataset_folder,
                train_sequences=data_config["split"]["train"],
                valid_sequences=data_config["split"]["valid"],
                test_sequences=data_config["split"]["test"],
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

# post processing
knn = KNN(model_params['post']['KNN']['params'], parser.get_n_classes())

# load model
model = RangeRet(model_params).to(device)
try:
    #load model
    model.load_state_dict(torch.load(model_path))
except:
    # load model from checkpoint
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.eval()

# print parameters
#for name, param in model.named_parameters():
    #print(f'{name} : {param}')

def infer(data_loader, to_original):
    # confusion matrix
    conf_matrix = np.zeros((20, 20), dtype=np.int64)

    # set of labels that appear in ground truth
    unique_gt = set()

    with torch.no_grad():
        #for i, data in enumerate(zip(range_images, labels_images)):
        for i, (in_vol, _, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, _) in tqdm(enumerate(data_loader), total=len(data_loader)):

            seq_folder = os.path.join(prediction_path, path_seq[0])
            # create sequence folder if it does not exists
            if not os.path.exists(seq_folder):
                os.mkdir(seq_folder)

            in_vol = in_vol.cuda()
            #proj_labels = proj_labels.cuda()
            p_x = p_x.cuda()
            p_y = p_y.cuda()

            #print(in_vol) # (B, H, W, C)

            outputs = model(in_vol) # input format (B, H, W, C)
            #outputs = model.forward_recurrent(in_vol) # input format (B, H, W, C)

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

            # populate confusion matrix
            #idxs = tuple(np.stack((proj_argmax.reshape(-1, 1).cpu().detach().numpy(), proj_labels.reshape(-1, 1).cpu().detach().numpy()), axis=0))
            #np.add.at(conf_matrix, idxs, 1)

            # put in original pointcloud using indexes or knn
            #unproj_argmax = proj_argmax[p_y, p_x]
            # use knn post processing
            unproj_argmax = knn(proj_range.cuda(), unproj_range.cuda(), proj_argmax, p_x, p_y)
            #print(unproj_argmax.shape)

            pred_np = unproj_argmax.cpu().detach().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            if split != 'test':
                # populate confusion matrix (iou between predicted point cloud and original labels)
                unproj_labels = unproj_labels.cpu().detach().numpy()
                idxs = tuple(np.stack((pred_np, unproj_labels.reshape(-1)), axis=0))
                np.add.at(conf_matrix, idxs, 1)

                unique_gt |= set(np.unique(unproj_labels))

            # back to original labels
            preds = to_original(pred_np)

            # prediction file path
            path = os.path.join(seq_folder, path_name[0])
            preds.tofile(path)

    # array of true-false
    label_presence = [index in unique_gt for index in range(0, 20)]
    label_presence[0] = False # outlier/unlabeled points
    #print(label_presence)

    # clean stats
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=1) - tp
    fn = conf_matrix.sum(axis=0) - tp

    intersection = tp
    union = tp + fp + fn + 1e-15

    iou = intersection / union
    iou_mean = (intersection[label_presence] / union[label_presence]).mean()

    return iou, iou_mean

print("INFERENCE:")

if split == 'train':
    iou, iou_mean = infer(data_loader=parser.get_train_set(), to_original=parser.to_original)
elif split == 'valid':
    iou, iou_mean = infer(data_loader=parser.get_valid_set(), to_original=parser.to_original)
elif split == 'test':
    iou, iou_mean = infer(data_loader=parser.get_test_set(), to_original=parser.to_original)
else:
    raise Exception('Invalid split selected')

for i in range(len(iou)):
    print('IoU class {} = {:.2%}'.format(i, iou[i]))

print('mIoU = {:.2%}'.format(iou_mean))