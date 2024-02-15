import os
import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cuda as cudnn
import torch.nn.functional as F
import numpy as np

from utils.avgmeter import AverageMeter
from utils.ioueval import iouEval
from utils.knn import KNN

from network.rangeret import RangeRet

from dataloader.kitti.parser import Parser

class User():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir, split):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.split = split

        # get the data
        self.parser = Parser(root=self.datadir,
                             train_sequences=self.DATA['split']['train'],
                             valid_sequences=self.DATA['split']['valid'],
                             test_sequences=self.DATA['split']['test'],
                             labels=self.DATA['labels'],
                             color_map=self.DATA['color_map'],
                             learning_map=self.DATA['learning_map'],
                             learning_map_inv=self.DATA['learning_map_inv'],
                             sensor=self.ARCH['dataset']['sensor'],
                             max_points=self.ARCH['dataset']['max_points'],
                             batch_size=1,
                             workers=self.ARCH['train']['workers'],
                             gt=True,
                             aug=False,
                             shuffle_train=False)
        
        # load model
        with torch.no_grad():
            self.model = RangeRet(self.ARCH['model_params'], self.parser.get_resolution(), self.parser.get_n_classes())

        try:
            #load model
            self.model.load_state_dict(torch.load(self.modeldir), strict=True)
            #self.model.load_state_dict(torch.load(self.modeldir, map_location=torch.device('cpu')), strict=True)
        except:
            # load model from checkpoint
            self.model.load_state_dict(torch.load(self.modeldir)['model_state_dict'], strict=True)

        # knn post processing
        self.post = None
        if self.ARCH['model_params']['post']['KNN']['use']:
            self.post = KNN(self.ARCH['model_params']['post']['KNN']['params'], self.parser.get_n_classes())

        # GPU
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Infering in device: ', self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

        # evaluation (ignore class 0)
        self.eval = True # set False to produce only predictions
        self.evaluator = iouEval(self.parser.get_n_classes(), self.device, [0])

    def infer(self):
        if self.split == 'train':
            # do train set
            acc, iou = self.infer_subset(loader=self.parser.get_train_set(),
                                         to_orig_fn=self.parser.to_original,
                                         evaluator=self.evaluator)
            print('Split: {} | acc: {:.2%} | iou: {:.2%}'.format(self.split, acc, iou))
        elif self.split == 'valid':
            acc, iou = self.infer_subset(loader=self.parser.get_valid_set(),
                                         to_orig_fn=self.parser.to_original,
                                         evaluator=self.evaluator)
            print('Split: {} | acc: {:.2%} | iou: {:.2%}'.format(self.split, acc, iou))
        elif self.split == 'test':
            self.infer_subset(loader=self.parser.get_train_set(),
                                         to_orig_fn=self.parser.to_original,
                                         evaluator=self.evaluator)
        else:
            raise SyntaxError('Invalid split chosen. Choose one of \'train\', \'valid\', \'test\'')

        print('Finished Infering')

    def infer_subset(self, loader, to_orig_fn, evaluator):
        acc = AverageMeter()
        iou = AverageMeter()
        # TODO save iou per class
        #iou_class = np.zeros(self.parser.get_n_classes())
        
        # switch to evaluation mode
        self.model.eval()

        evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            for i, (proj_in, _, _, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in tqdm(enumerate(loader), total=len(loader)):
                # first cut to rela size (batch size one allows it)
                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.eval and self.split != 'test':
                    unproj_labels = unproj_labels[0, :npoints]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()

                proj_output = self.model(proj_in)
                predictions = proj_output.permute(0, 3, 1, 2)
                proj_argmax = predictions[0].argmax(dim=0)

                if self.post:
                    # knn post processing
                    unproj_argmax = self.post(proj_range,
                                              unproj_range,
                                              proj_argmax,
                                              p_x,
                                              p_y)
                else:
                    # put in original pointcloud using indexes
                    unproj_argmax = proj_argmax[p_y, p_x]

                # save scan
                # get the first scan in batch and project scan
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # compute metrics
                if self.eval and self.split != 'test':
                    evaluator.addBatch(pred_np, unproj_labels)
                    accuracy = evaluator.getacc()
                    jaccard, class_jaccard = evaluator.getIoUMissingClass()
                    acc.update(accuracy.item(), 1)
                    iou.update(jaccard.item(), 1)

                # map to original label
                pred_np = to_orig_fn(pred_np)

                # save scan
                path = os.path.join(self.logdir, "sequences",
                                    path_seq, "predictions", path_name)
                pred_np.tofile(path)

        if self.split != 'train':
            # TODO return also iou per class
            return acc.avg, iou.avg        