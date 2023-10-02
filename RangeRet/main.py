# main file

import os
import sys
import yaml
import torch
import numpy as np
from torch import nn
from sklearn.feature_extraction import image

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
sequence_folder = sys.argv[1]
velodyne_folder = os.path.join(sequence_folder, 'velodyne')
labels_folder = os.path.join(sequence_folder, 'labels')

# config path (label mapping)
config = load_yaml('./config/label_mapping/semantic-kitti.yaml')
learning_map = config['learning_map']

range_images = []
labels_images = []

def project_scan(scan, labels):
    ### Create Range Image
    H = 64
    W = 1024

    fov_up = 3.0 / 180.0 * np.pi
    fov_down = -25 / 180.0 * np.pi

    # range image with (x, y, z, depth, remission) features
    range_image = np.zeros((H, W, 5), np.float32)
    label_image = np.zeros((H, W, 1), np.uint32)

    # Compute the range of the point cloud
    r = np.sqrt(np.sum(np.power(scan[:, :3], 2), axis=1))

    # Compute the polar and azimuthal angles of the points
    pitch = np.arcsin(scan[:, 2] / r)
    yaw = np.arctan2(scan[:, 1], scan[:, 0])

    fov = fov_up + np.abs(fov_down)

    # create range image
    for i, p in enumerate(scan):
        # print(pitch[i], fov_down, fov)
        u = H * (1 - ((pitch[i] + np.abs(fov_down)) / fov))
        v = W * (0.5 * ((yaw[i] / (np.pi / 2)) + 1.0))

        # round to the nearest integer
        u = int(np.round(min(u, H - 1)))
        v = int(np.round(min(v, W - 1)))

        u = max(0, u)
        v = max(0, v)

        # print(u, v)

        # range image
        range_image[u, v, :4] = p
        range_image[u, v, 4] = r[i]

        # label image
        label_image[u, v] = labels[i]
    
    return range_image, label_image

for file in os.listdir(velodyne_folder):
    scan_file = os.path.join(velodyne_folder, file)
    scan = np.fromfile(scan_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    #print(f'Number of points in sample scan: {scan.shape[0]}')

    # read labels
    label_file = os.path.join(labels_folder, file.replace('bin', 'label'))
    labels = np.fromfile(label_file, dtype=np.uint32)
    labels = labels.reshape(-1)
    #print(f'Number of labels in scan: {labels.shape[0]}')
    labels = labels & 0xFFFF
    labels = np.vectorize(learning_map.__getitem__)(labels)

    assert scan.shape[0] == labels.shape[0], 'Different number of points'

    range_proj, label_proj = project_scan(scan, labels)

    range_images.append(range_proj)
    labels_images.append(label_proj)

range_images = np.array(range_images)
labels_images = np.array(labels_images, dtype=np.int64)

labels_images = torch.from_numpy(labels_images).to(device)

# print(range_images.shape)
# print(labels_images.shape)

print('Upload range images')

model = RangeRet().to(device)

#loss_fn = nn.NLLLoss()
loss_fn = nn.NLLLoss(ignore_index=0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-8)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(zip(range_images, labels_images)):
        patches = image.extract_patches_2d(data[0], (4, 4), max_patches=4096)
        #print(patches.shape)

        inputs = patches.reshape(patches.shape[0], patches.shape[1] * patches.shape[2], patches.shape[3]) # shape = (4069, 16, 5)
        #print(inputs.shape)

        inputs = torch.from_numpy(inputs).to(device)

        # for visionEmbedding
        #inputs = torch.from_numpy(data[0]).to(device)
        #inputs = inputs.reshape(1, 64, 1024, 5)
        #inputs = inputs.permute(0, 3, 1, 2)

        optimizer.zero_grad()

        outputs = model(inputs)

        #print('outputs shape: ', outputs.shape)
        #print('labels shape: ', labels_images[i].shape)

        outputs = outputs.reshape(1, 64, 1024, 20)

        predictions = outputs.permute(0, 3, 1, 2)
        gt = data[1].permute(2, 0, 1)
        #print(torch.argmax(predictions, dim=1))
        #print(gt)

        #print('predictions shape ', predictions.shape)
        #print('labels shape ', gt.shape)

        loss = loss_fn(torch.log(predictions.clamp(min=1e-8)), gt.cuda(non_blocking=True))
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return loss

epoch_number = 0

EPOCHS = 50

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, None)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    #with torch.no_grad():
    #    for i, vdata in enumerate(validation_loader):
    #        vinputs, vlabels = vdata
    #        voutputs = model(vinputs)
    #        vloss = loss_fn(voutputs, vlabels)
    #        running_vloss += vloss

    print('LOSS train {} '.format(avg_loss))

    epoch_number += 1