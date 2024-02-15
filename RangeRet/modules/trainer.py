import os
import sys
import time
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cuda as cudnn
import torch.nn.functional as F

from utils.avgmeter import AverageMeter
from utils.ioueval import iouEval
from utils.lovasz_loss import Lovasz_loss
from utils.focal_loss import FocalLoss

from network.rangeret import RangeRet

from dataloader.kitti.parser import Parser

class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.path = path

        # get data
        self.parser = Parser(root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=None,
            labels=self.DATA["labels"],
            color_map=self.DATA["color_map"],
            learning_map=self.DATA["learning_map"],
            learning_map_inv=self.DATA["learning_map_inv"],
            sensor=self.ARCH["dataset"]["sensor"],
            max_points=self.ARCH["dataset"]["max_points"],
            batch_size=self.ARCH["train"]["batch_size"],
            workers=self.ARCH["train"]["workers"],
            gt=True,
            aug=True,
            shuffle_train=True)

        # weights for loss and bias
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in self.DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)   # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if self.DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        with torch.no_grad():
            self.model = RangeRet(self.ARCH['model_params'], self.parser.get_resolution(), self.parser.get_n_classes())

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of parameters {num_params/1000000} M')

        # GPU
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Training in device: {self.device}')

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()

        # TODO: implement multi-gpu support
        #if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        #    self.model = nn.DataParallel(self.model)  # spread in gpus
        #    self.model = convert_model(self.model).cuda()  # sync batchnorm
        #    self.model_single = self.model.module  # single model to get weight names
        #    self.multi_gpu = True
        #    self.n_gpus = torch.cuda.device_count()

        # Losses
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, weight=self.loss_w).to(self.device)
        self.lovasz = Lovasz_loss(ignore=0).to(self.device)
        self.focal = FocalLoss(gamma=2.0, ignore_index=0).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.05, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, steps_per_epoch=len(self.parser.get_train_set()), epochs=self.ARCH['train']['epochs'], pct_start=0.3)

        print(f'Optimizer: {self.optimizer}')
        print(f'Scheduler: {self.scheduler}')

        # TODO add pretrained model config

    def train(self):
        # accuracy and IoU stuff
        best_train_iou = 0.0
        best_val_iou = 0.0

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(), self.device, self.ignore_class)

        #log_data = [['train_loss', 'val_loss', 'train_iou', 'val_iou']]
        log_data = []

        # trian for n epochs
        for epoch in range(self.ARCH['train']['epochs']):
            print(f'EPOCH {epoch+1}:')
            # train for 1 epoch
            acc, iou, loss = self.train_epoch(train_loader=self.parser.get_train_set(),
                                              model=self.model,
                                              criterion=self.criterion,
                                              optimizer=self.optimizer,
                                              epoch=epoch,
                                              evaluator=self.evaluator,
                                              scheduler=self.scheduler)

            print('Train | acc: {:.2%} | mIoU: {:.2%} | loss: {:.5}'.format(acc, iou, loss))

            # update best iou and save checkpoint
            if iou > best_train_iou:
                print('Best mIoU in training set so far, save model!')
                best_train_iou = iou
                # TODO save checkpoint
                #torch.save(self.model.state_dict(), f"{ARCH['model_architecture']}-model.pt")

            if epoch % self.ARCH['train']['report_epoch'] == 0:
                # evaluate on validation set
                val_acc, val_iou, val_loss = self.validate(val_loader=self.parser.get_valid_set(),
                                               model=self.model,
                                               criterion=self.criterion,
                                               epoch=epoch,
                                               evaluator=self.evaluator)

                print('Validation | acc: {:.2%} | mIoU: {:.2%} | loss: {:.5}'.format(val_acc, val_iou, val_loss))

                if val_iou > best_val_iou:
                    print('Best mIoU in validation so far, model saved!')
                    best_val_iou = val_iou
                    # TODO save the weights
                    #torch.save(self.model.state_dict(), f"{ARCH['model_architecture']}-model.pt")
            
            # update log
            log_data.append((acc, iou, loss, val_acc, val_iou, val_loss))

        # log data
        log_data = np.array(log_data, dtype=np.float32)
        np.savetxt(os.path.join(self.logdir, 'training_log.txt'), log_data, fmt='%f')

        torch.save(self.model.state_dict(), os.path.join(self.logdir, f"{self.ARCH['model_params']['model_architecture']}-model.pt"))

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

        if self.gpu:
            torch.cuda.empty_cache()

        model.train()

        for i, (in_vol, _, proj_labels, unproj_labels, _, _, p_x, p_y, proj_range, unproj_range, _, _, _, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            
            if self.gpu:
                in_vol = in_vol.cuda()
                proj_labels = proj_labels.cuda(non_blocking=True)

            output = model(in_vol)

            predictions = output.permute(0, 3, 1, 2)

            # compute loss
            # TODO use predictions from each stage ?
            ce_loss = criterion(predictions, proj_labels.long())
            lovasz_loss = self.lovasz(F.softmax(predictions, dim=1), proj_labels.long())
            focal_loss = self.focal(predictions, proj_labels.long())

            loss = ce_loss + focal_loss + lovasz_loss

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                evaluator.reset()
                argmax = predictions.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoUMissingClass()
            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            scheduler.step()

        return acc.avg, iou.avg, losses.avg

    def validate(self, val_loader, model, criterion, epoch, evaluator):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

        if self.gpu:
            torch.cuda.empty_cache()

        model.eval()
        evaluator.reset()

        with torch.no_grad():
            for i, (in_vol, _, proj_labels, unproj_labels, _, _, p_x, p_y, proj_range, unproj_range, _, _, _, _, _) in tqdm(enumerate(val_loader), total=len(val_loader)):

                if self.gpu:
                    in_vol = in_vol.cuda()
                    proj_labels = proj_labels.cuda(non_blocking=True)

                output = model(in_vol)

                predictions = output.permute(0, 3, 1, 2)

                # compute loss
                # TODO use predictions from each stage ?
                ce_loss = criterion(predictions, proj_labels.long())
                lovasz_loss = self.lovasz(F.softmax(predictions, dim=1), proj_labels.long())
                focal_loss = self.focal(predictions, proj_labels.long())

                loss = ce_loss + focal_loss + lovasz_loss

                argmax = predictions.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.item(), in_vol.size(0))

                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoUMissingClass()
                acc.update(accuracy.item(), in_vol.size(0))
                iou.update(jaccard.item(), in_vol.size(0))

        return acc.avg, iou.avg, losses.avg