# main file
# Usage python3 main.py <config_path> <data_path>

# profiler
import pstats
import cProfile

import os
import sys
import yaml
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F

from utils.knn import KNN
from dataloader.pandaset.parser import Parser
from utils.lovasz_loss import Lovasz_loss
from utils.focal_loss import FocalLoss
from utils.lion_optimizer import Lion

from network.rangeret import RangeRet

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

def seed_everything(seed=1064):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
    print(f'Using seed = {seed}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything()

# input data
config_path = sys.argv[1]
dataset_folder = sys.argv[2]
save_checkpoint = True
load_checkpoint = False

# log folder
log_dir = 'log/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# checkpoints folder
checkpoints_dir = 'checkpoints/'
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

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
                aug=True,
                shuffle_train=True)

model = RangeRet(model_params).to(device)
#model.load_state_dict(torch.load('rangeret-ocycle.pt'))

# count number of parameters
print('Total params: ', sum(p.numel() for p in model.parameters()))

# weights for loss and bias
epsilon_w = config["train_params"]["epsilon_w"]
content = torch.zeros(parser.get_n_classes(), dtype=torch.float)
for cl, freq in data_config["content"].items():
  x_cl = parser.to_xentropy(cl)  # map actual class to xentropy class
  content[x_cl] += freq
loss_w = 1 / (content + epsilon_w)   # get weights
for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
  if data_config["learning_ignore"][x_cl]:
    # don't weigh
    loss_w[x_cl] = 0
print("Loss weights from content: ", loss_w.data)

# TODO use focal loss, lovasz loss
ce_criterion = nn.CrossEntropyLoss(ignore_index=0, weight=loss_w).to(device)
lovasz_criterion = Lovasz_loss(ignore=0).to(device)
focal_criterion = FocalLoss(gamma=2.0, ignore_index=0).to(device) # TODO set gamma=2.0
#nll_criterion = nn.NLLLoss(ignore_index=0).to(device)

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, eps=1e-8)
#optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=5e-2)
# scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(parser.get_train_set()), epochs=config['train_params']['num_epochs'], pct_start=0.02)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train_params']['num_epochs'] - 4, eta_min=1e-5)

# post processing
#knn = KNN(model_params['post']['KNN']['params'], parser.get_n_classes())

def train_one_epoch(train_loader, epoch_index):
    torch.cuda.empty_cache()
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)

    running_loss = 0.

    # confusion matrix
    conf_matrix = np.zeros((20, 20), dtype=np.int64)

    #for i, data in enumerate(zip(range_images, labels_images)):
    for i, (in_vol, _, proj_labels, unproj_labels, _, _, p_x, p_y, proj_range, unproj_range, _, _, _, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):

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

        proj_argmax = predictions.argmax(dim=1)
        #print(proj_argmax)
        #print(proj_labels)

        #np.savetxt('pred.txt', proj_argmax[0].cpu().detach().numpy(), fmt='%d')
        #np.savetxt('labels.txt', proj_labels[0].cpu().detach().numpy(), fmt='%d')

        #print('predictions shape ', predictions.shape)
        #print('labels shape ', gt.shape)

        # loss between range images
        ce_loss = ce_criterion(predictions, proj_labels.cuda(non_blocking=True).long())
        lovasz_loss = lovasz_criterion(F.softmax(predictions, dim=1), proj_labels.cuda(non_blocking=True).long())
        focal_loss = focal_criterion(predictions, proj_labels.cuda(non_blocking=True).long())

        # loss between point clouds
        #preds = outputs[0][p_y, p_x]
        #print(unproj_labels.shape)
        #ce_loss = ce_criterion(preds.permute(0, 2, 1), unproj_labels.cuda(non_blocking=True).long())
        #lovasz_loss = lovasz_criterion(preds.permute(0, 2, 1), unproj_labels.cuda(non_blocking=True).long())
        #focal_loss = focal_criterion(preds.permute(0, 2, 1), unproj_labels.cuda(non_blocking=True).long())

        loss = ce_loss + focal_loss + lovasz_loss

        #ce_loss.backward()
        #lovasz_loss.backward()
        #focal_loss.backward()
        loss.backward()

        optimizer.step()
        scheduler.step()

        # Gather data and report
        #running_loss += ce_loss.item()
        #running_loss += lovasz_loss.item()
        #running_loss += focal_loss.item()
        running_loss += loss.item()

        # populate confusion matrix
        idxs = tuple(np.stack((proj_argmax.reshape(-1, 1).cpu().detach().numpy(), proj_labels.reshape(-1, 1).cpu().detach().numpy()), axis=0))
        np.add.at(conf_matrix, idxs, 1)

        # TODO put in original pointcloud using indexes and compute loss between whole point cloud labels 
        #unproj_argmax = proj_argmax[p_y, p_x]
        #unproj_argmax = knn(proj_range, unproj_range, proj_argmax, p_x, p_y)
        #print(unproj_argmax.shape)
        #print(unproj_labels.shape)

        #pred_np = unproj_argmax.cpu().detach().numpy()
        #pred_np = pred_np.reshape((-1)).astype(np.int32)

        # populate confusion matrix (iou between predicted point cloud and original labels)
        #idxs = tuple(np.stack((pred_np, unproj_labels.cpu().detach().numpy().reshape(-1)), axis=0))
        #np.add.at(conf_matrix, idxs, 1)

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

    return running_loss / len(train_loader), iou_mean

def validate(val_loader):
    torch.cuda.empty_cache()
    
    model.eval()

    val_loss = 0.

    # confusion matrix
    conf_matrix = np.zeros((20, 20), dtype=np.int64)

    with torch.no_grad():
        #for i, data in enumerate(zip(range_images, labels_images)):
        for i, (in_vol, _, proj_labels, unproj_labels, _, _, p_x, p_y, proj_range, unproj_range, _, _, _, _, _) in tqdm(enumerate(val_loader), total=len(val_loader)):

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

            proj_argmax = predictions.argmax(dim=1)
            #print(proj_argmax)
            #print(proj_labels)

            #np.savetxt('pred.txt', proj_argmax[0].cpu().detach().numpy(), fmt='%d')
            #np.savetxt('labels.txt', proj_labels[0].cpu().detach().numpy(), fmt='%d')

            #print('predictions shape ', predictions.shape)
            #print('labels shape ', gt.shape)

            # loss between images
            ce_loss = ce_criterion(predictions, proj_labels.cuda(non_blocking=True).long())
            lovasz_loss = lovasz_criterion(F.softmax(predictions, dim=1), proj_labels.cuda(non_blocking=True).long())
            focal_loss = focal_criterion(predictions, proj_labels.cuda(non_blocking=True).long())

            #preds = outputs[0][p_y, p_x]

            #loss = loss_fn(torch.log(predictions.clamp(min=1e-8)), gt.cuda(non_blocking=True))
            #ce_loss = ce_criterion(preds.permute(0, 2, 1), unproj_labels.cuda(non_blocking=True).long())
            #lovasz_loss = lovasz_criterion(preds.permute(0, 2, 1), unproj_labels.cuda(non_blocking=True).long())
            #focal_loss = focal_criterion(preds.permute(0, 2, 1), unproj_labels.cuda(non_blocking=True).long())

            loss = ce_loss + focal_loss + lovasz_loss

            #val_loss += ce_loss.item()
            #val_loss += lovasz_loss.item()
            #val_loss += focal_loss.item()
            val_loss += loss.item()

            # populate confusion matrix
            idxs = tuple(np.stack((proj_argmax.reshape(-1, 1).cpu().detach().numpy(), proj_labels.reshape(-1, 1).cpu().detach().numpy()), axis=0))
            np.add.at(conf_matrix, idxs, 1)

            # put in original pointcloud using indexes or knn
            #unproj_argmax = proj_argmax[p_y, p_x]
            #unproj_argmax = knn(proj_range, unproj_range, proj_argmax, p_x, p_y)
            #print(unproj_argmax.shape)

            #pred_np = unproj_argmax.cpu().detach().numpy()
            #pred_np = pred_np.reshape((-1)).astype(np.int32)

            # populate confusion matrix (iou between predicted point cloud and original labels)
            #idxs = tuple(np.stack((pred_np, unproj_labels.cpu().detach().numpy().reshape(-1)), axis=0))
            #np.add.at(conf_matrix, idxs, 1)

            #np.savetxt('pc_predicitons.txt', pred_np)

    # clean stats
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=1) - tp
    fn = conf_matrix.sum(axis=0) - tp

    intersection = tp
    union = tp + fp + fn + 1e-15

    iou = intersection / union
    iou_mean = (intersection / union).mean()

    return val_loss / len(val_loader), iou_mean

start_epoch = 0
EPOCHS = config['train_params']['num_epochs']

#load checkpoint
if load_checkpoint:
    checkpoint = torch.load('checkpoints/checkpoint51-64epochs.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] - 1
    loss = checkpoint['loss']
    print(f'CHECKPOINT LOADED | epoch = {start_epoch} | loss = {loss}')

#log_data = [['train_loss', 'val_loss', 'train_iou', 'val_iou']]
log_data = []

for epoch in range(start_epoch, EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    #cProfile.run("train_one_epoch(train_loader=parser.get_train_set(), epoch_index=epoch_number)", "my_func_stats")
    #p = pstats.Stats("my_func_stats")
    #p.sort_stats("cumulative").print_stats()
    
    avg_loss, iou_mean = train_one_epoch(train_loader=parser.get_train_set(), epoch_index=epoch)

    print('TRAIN Loss = {} | mIoU = {:.2%}'.format(avg_loss, iou_mean))

    val_loss, val_iou = validate(val_loader=parser.get_valid_set())

    print('VALIDATION Loss = {} | mIoU = {:.2%}'.format(val_loss, val_iou))

    # TODO add log for train/valid loss and mIoU
    log_data.append([avg_loss, val_loss, iou_mean, val_iou])

    # save model checkpoint
    if save_checkpoint:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }, 'checkpoints/model-checkpoint.pt')

# log data
log_data = np.array(log_data, dtype=np.float32)
np.savetxt(os.path.join(log_dir, datetime.today().strftime('%Y-%m-%d %H:%M:%S.txt')), log_data, fmt='%f')

# save model
torch.save(model.state_dict(), f"{model_params['model_architecture']}-model.pt")