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
import matplotlib.pyplot as plt

from utils.avgmeter import AverageMeter
from utils.ioueval import iouEval
from utils.lovasz_loss import Lovasz_loss
from utils.boundary_loss import BoundaryLoss

from torch.utils.tensorboard import SummaryWriter

from utils.sync_batchnorm.batchnorm import convert_model
from utils.scheduler import WarmupCosine, WarmupCosineLR

from network.rangeret import RangeRet

from dataloader.rangeaug import RangeAugmentation

class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, checkpoint=None, pretrained=None, fp16=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.checkpoint = checkpoint
        self.pretrained = pretrained
        self.fp16 = fp16

        # get data
        if self.ARCH['dataset']['pc_dataset_type'] == 'SemanticKITTI':
            from dataloader.kitti.parser import Parser
            self.dataset_type = 'kitti'
        elif self.ARCH['dataset']['pc_dataset_type'] == 'PandaSet':
            from dataloader.pandaset.parser import Parser
            self.dataset_type = 'pandaset'
        elif self.ARCH['dataset']['pc_dataset_type'] == 'SemanticPOSS':
            from dataloader.poss.parser import Parser
            self.dataset_type = 'poss'
        else:
            raise ValueError(f"Dataset type {self.ARCH['dataset']['pc_dataset_type']} not supported")
        
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
            self.model = RangeRet(self.ARCH['model_params'],
                                  self.parser.get_resolution(),
                                  self.parser.get_n_classes())

        # print details of the model
        # for name, param in self.model.named_parameters():
        #     print(f"Layer: {name}, Parameters: {param.numel()}")

        print(f'Backbone: {self.ARCH["model_params"]["backbone"]}')

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

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = convert_model(self.model)  # sync batchnorm
            self.model = nn.DataParallel(self.model).cuda()  # spread in gpus
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()

        # Losses
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ARCH['dataset']['ignore_label'], weight=self.loss_w).to(self.device)
        self.lovasz = Lovasz_loss(ignore=self.ARCH['dataset']['ignore_label']).to(self.device)
        self.bd = BoundaryLoss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()
            self.lovasz = nn.DataParallel(self.lovasz).cuda()
            #self.focal = nn.DataParallel(self.focal).cuda()
            self.bd = nn.DataParallel(self.bd).cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.ARCH['train']['learning_rate'],
                                           weight_decay=self.ARCH['train']['weight_decay'],
                                           eps=1e-8)
        
        # Scheduler
        if self.ARCH['train']['scheduler']['name'] == 'OneCycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=self.ARCH['train']['scheduler']['max_lr'],
                                                             steps_per_epoch=len(self.parser.get_train_set()),
                                                             epochs=self.ARCH['train']['epochs'],
                                                             pct_start=self.ARCH['train']['scheduler']['pct_start'])
        elif self.ARCH['train']['scheduler']['name'] == 'WarmupCosine':
            self.scheduler = WarmupCosineLR(self.optimizer,
                                            lr=self.ARCH['train']['learning_rate'],
                                            warmup_steps=5 * self.parser.get_train_size(), # TODO set warmup epochs in config file
                                            momentum=0.9,
                                            max_steps=(self.ARCH['train']['epochs'] - 5) * self.parser.get_train_size())
        else:
            # setup a simple scheduler
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=1)

        print(f'Optimizer: {self.optimizer}')
        print(f'Scheduler: {self.scheduler}')

        # grad scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # range augmentation (mix, union, paste, shift)
        if self.ARCH['train']['range_aug']:
            self.range_aug = RangeAugmentation(dataset=self.dataset_type)

        # Checkpoint model config
        if self.checkpoint is not None:
            try:
                self.model.load_state_dict(torch.load(self.checkpoint))
                print(f'Checkpoint loaded from {self.checkpoint}')
            except:
                print(f'Error loading checkpoint from {self.checkpoint}')
        
        # Pretrained RetNet model
        if self.pretrained is not None:
            try:
                self.model.backbone.load_state_dict(torch.load(self.pretrained))
                print(f'Pre-trained RetNet loaded from {self.pretrained}')
            except:
                print(f'Error loading RetNet from {self.pretrained}')

        # tensorboard
        self.writer_train = SummaryWriter(log_dir=self.logdir + "/tensorboard/train/", flush_secs=30)
        self.writer_val = SummaryWriter(log_dir=self.logdir + "/tensorboard/val/", flush_secs=30)

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
            print(f'EPOCH {epoch+1}/{self.ARCH["train"]["epochs"]}')
            # train for 1 epoch
            acc, iou, loss = self.train_epoch(train_loader=self.parser.get_train_set(),
                                              model=self.model,
                                              criterion=self.criterion,
                                              optimizer=self.optimizer,
                                              epoch=epoch,
                                              show_scans=self.ARCH['train']['show_scans'],
                                              color_fn=self.parser.to_color,
                                              evaluator=self.evaluator,
                                              scheduler=self.scheduler)

            print('Train | acc: {:.2%} | mIoU: {:.2%} | loss: {:.5}'.format(acc, iou, loss))

            # update best iou and save checkpoint
            if iou > best_train_iou:
                print('Best mIoU in training set so far!')
                best_train_iou = iou
                # TODO save checkpoint
                #torch.save(self.model.state_dict(), f"{ARCH['model_architecture']}-model.pt")

            if epoch % self.ARCH['train']['report_epoch'] == 0:
                # evaluate on validation set
                val_acc, val_iou, val_loss = self.validate(val_loader=self.parser.get_valid_set(),
                                               model=self.model,
                                               criterion=self.criterion,
                                               epoch=epoch,
                                               show_scans=self.ARCH['train']['show_scans'],
                                               color_fn=self.parser.to_color,
                                               evaluator=self.evaluator)

                print('Validation | acc: {:.2%} | mIoU: {:.2%} | loss: {:.5}'.format(val_acc, val_iou, val_loss))

                if val_iou > best_val_iou:
                    print('Best mIoU in validation so far, model saved!')
                    best_val_iou = val_iou
                    # TODO save the weights
                    torch.save(self.model_single.state_dict(), os.path.join(self.logdir, f"{self.ARCH['model_params']['model_architecture']}-best.pt"))
            
            # update log
            log_data.append((acc, iou, loss, val_acc, val_iou, val_loss))

        # log data
        log_data = np.array(log_data, dtype=np.float32)
        np.savetxt(os.path.join(self.logdir, 'training_log.txt'), log_data, fmt='%f')

        torch.save(self.model_single.state_dict(), os.path.join(self.logdir, f"{self.ARCH['model_params']['model_architecture']}-last.pt"))

        print('Finished Training')

        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, show_scans, color_fn, evaluator, scheduler):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

        if self.gpu:
            torch.cuda.empty_cache()

        model.train()

        for i, (in_vol, proj_mask, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda(non_blocking=True).long()

            if self.ARCH['train']['range_aug']:
                in_vol, proj_labels = self.range_aug(in_vol, proj_labels, proj_mask)

            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs = model(in_vol)

                predictions = outputs.permute(0, 3, 1, 2)

                # compute loss
                ce_loss = criterion(predictions, proj_labels)
                lovasz_loss = self.lovasz(F.softmax(predictions, dim=1), proj_labels)
                bd_loss = self.bd(F.softmax(predictions, dim=1), proj_labels)

                loss = ce_loss + bd_loss + 1.5 * lovasz_loss

            self.scaler.scale(loss).backward()

            self.scaler.step(optimizer)
            self.scaler.update()

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

            if show_scans and i % 10 == 0:
                # show the images
                pred_img = color_fn(argmax[0].cpu().numpy())
                gt_img = color_fn(proj_labels[0].cpu().numpy())
                
                pred_img = pred_img[:, :, ::-1]
                gt_img = gt_img[:, :, ::-1]
                
                fig, axs = plt.subplots(2, 1, figsize=(15, 5))
                axs[0].imshow(pred_img)
                axs[0].set_title('Prediction')
                axs[0].axis('off')
                axs[1].imshow(gt_img)
                axs[1].set_title('Ground Truth')
                axs[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.logdir, f'training_scan.png'))
                plt.close()

            # tensorboard
            if i % 10 == 0 or i == len(train_loader) - 1:
                header = "Train"
                step = epoch * len(train_loader) + i
                self.writer_train.add_scalar(header + '/loss', losses.avg, step)
                self.writer_train.add_scalar(header + '/accuracy', acc.avg, step)
                self.writer_train.add_scalar(header + '/mIoU', iou.avg, step)
                self.writer_train.add_scalar(header + "/lr", self.optimizer.param_groups[0]["lr"], step)
                self.writer_train.add_scalar(header + "/ce_loss", ce_loss.item(), step)
                self.writer_train.add_scalar(header + "/lovasz_loss", lovasz_loss.item(), step)
                #self.writer_train.add_scalar(header + "/focal_loss", focal_loss.item(), step)
                self.writer_train.add_scalar(header + "/bd_loss", bd_loss.item(), step)

        return acc.avg, iou.avg, losses.avg

    def validate(self, val_loader, model, criterion, epoch, show_scans, color_fn, evaluator):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

        if self.gpu:
            torch.cuda.empty_cache()

        model.eval()
        evaluator.reset()

        with torch.no_grad():
            for i, (in_vol, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(val_loader), total=len(val_loader)):

                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = model(in_vol)
                    predictions = outputs.permute(0, 3, 1, 2)

                    # compute loss
                    ce_loss = criterion(predictions, proj_labels)
                    lovasz_loss = self.lovasz(F.softmax(predictions, dim=1), proj_labels)
                    bd_loss = self.bd(F.softmax(predictions, dim=1), proj_labels)

                    loss = ce_loss + bd_loss + 1.5 * lovasz_loss

                argmax = predictions.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.item(), in_vol.size(0))

                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoUMissingClass()
                acc.update(accuracy.item(), in_vol.size(0))
                iou.update(jaccard.item(), in_vol.size(0))

                if show_scans and i % 10 == 0:
                    # show the images
                    pred_img = color_fn(argmax[0].cpu().numpy())
                    gt_img = color_fn(proj_labels[0].cpu().numpy())

                    pred_img = pred_img[:, :, ::-1]
                    gt_img = gt_img[:, :, ::-1]

                    fig, axs = plt.subplots(2, 1, figsize=(15, 5))
                    axs[0].imshow(pred_img)
                    axs[0].set_title('Predictions')
                    axs[0].axis('off')
                    axs[1].imshow(gt_img)
                    axs[1].set_title('Ground Truth')
                    axs[1].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.logdir, f'validation_scan.png'))
                    plt.close()
            
            # tensorboard
            header = "Validation"
            step = epoch
            self.writer_val.add_scalar(header + '/loss', losses.avg, step)
            self.writer_val.add_scalar(header + '/accuracy', acc.avg, step)
            self.writer_val.add_scalar(header + '/mIoU', iou.avg, step)
            self.writer_val.add_scalar(header + "/ce_loss", ce_loss.item(), step)
            self.writer_val.add_scalar(header + "/lovasz_loss", lovasz_loss.item(), step)
            #self.writer_val.add_scalar(header + "/focal_loss", focal_loss.item(), step)
            self.writer_val.add_scalar(header + "/bd_loss", bd_loss.item(), step)

        return acc.avg, iou.avg, losses.avg