import sys
import datetime
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from fastai.vision.data import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.callbacks.tracker import *
from fastai.basic_train import *

# @t_sanf @DrSHarmon

class VisionMulticlass:
    '''class to save hyperparameters while training models using fastai library functionality'''

    def __init__(self):
        self.imagedir=''
        self.outdir=''
        self.testPath=os.path.join(self.imagedir,'test')
        self.model_name='bot2'
        self.tr_name='train'
        self.val_name='val'
        self.arch='resnet34'
        self.img_sz=80
        self.lr=0.005
        self.lr_range=slice(1e-11, 1e-5)
        self.bs=128
        self.device=0
        self.dc_e=20
        self.all_e=1000 #all epochs
        self.lighting=0.05
        self.rotate=45
        self.weightedloss=False
        self.early_stopping=True
        self.unfreeze=True
        self.weights = [10,5,2.5,1]
        self.model_dict={'resnet18':[models.resnet18],'resnet34':[models.resnet34],'resnet50':[models.resnet50],
            'resnet101':[models.resnet101],'resnet152':[models.resnet152],'vgg16_bn':[models.vgg16_bn],'densenet161':[models.densenet161]}

        #retraining
        self.retrain=False
        self.retraindir = '/data/Stephanie_Harmon/bladder_path/all_pts/classification_042519/All_Others/10x/training_log'
        self.save_model_name='10x_fullset_fortytwo_final_layers_tuned_04302019-0824'

    def load_jpg_from_folder(self):
        '''
        expects a path to a base folder with multiple subfolders including 'training', 'testing', etc
        :param path:
        :return: databunch
        '''


        tfms = get_transforms(flip_vert=True,max_rotate=self.rotate,max_warp=0.05,max_lighting = self.lighting,p_lighting=0.9,p_affine=0.5)

        data = (ImageList.from_folder(self.imagedir)
                .split_by_folder(train=self.tr_name, valid=self.val_name)
                .label_from_folder()
                .transform(tfms, size=self.img_sz)
                .databunch(bs=self.bs)
                .normalize())
        return data

    def train(self):
        '''
        trains a resnet with the parameters listed above
        :return:
        '''

        torch.cuda.set_device(self.device)

        self.make_filestructure()

        data = self.load_jpg_from_folder()
        print('data loaded')
        print('classes in this dataset are {}'.format(data.classes))


        w = torch.cuda.FloatTensor(self.weights)
        learn = cnn_learner(data, self.model_dict[self.arch][0],
                            metrics=[error_rate],
                            callback_fns=[ShowGraph,
                                          partial(SaveModelCallback, monitor='valid_loss', mode='auto',name=self.model_name),
                                          partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.001,patience=100)],
                            wd=0.1,
                            loss_func=LabelSmoothingCrossEntropy()).mixup()

        self.trainblock(learner=learn)
        save_name = self.model_name + "_" + str(datetime.datetime.now().strftime("%m%d%Y-%H%M")) + '.pkl'

        #save figures
        self.save_figures(learner=learn)

        # save hyperparameters
        self.save_hyperparameters(filename='hyperparameters')

        return save_name

####################
# helper functions #
####################

    def trainblock(self,learner):
        '''basic block to train network
        a fastai learner will need tobe defined ebfore the model is trained
        '''

        learn=learner
        learn.fit_one_cycle(self.dc_e, max_lr=self.lr)
        learn.save(os.path.join(self.outdir, 'saved_models', self.model_name + "_" + 'final_layers_tuned_' + str(
            datetime.datetime.now().strftime("%m%d%Y-%H%M"))))
        learn.export(os.path.join(self.outdir, 'exported_models', self.model_name + "_" + str(
            datetime.datetime.now().strftime("%m%d%Y-%H%M") + '.pkl')))

        # loop to train if unfreezing is desired
        if self.unfreeze == True:
            print("unfreezing and retraining")
            learn.unfreeze()
            learn.fit_one_cycle(self.all_e, max_lr=self.lr_range)
            learn.save(
                os.path.join(self.outdir, 'saved_models', self.model_name + "_" + 'all_layers_trained_' + str(
                    datetime.datetime.now().strftime("%m%d%Y-%H%M"))))
            learn.export(os.path.join(self.outdir, 'exported_models', self.model_name + "_" + str(
                datetime.datetime.now().strftime("%m%d%Y-%H%M"))))

    def save_figures(self,learner):
        interp = ClassificationInterpretation.from_learner(learner)
        losses, idxs = interp.top_losses()
        interp.plot_confusion_matrix(figsize=(12, 12),dpi=60)
        plt.savefig(os.path.join(self.outdir, 'confusion_matrix',
                                 self.model_name + '_' + str(datetime.datetime.now().strftime("%m%d%Y-%H%M"))))
        interp.plot_top_losses(9, figsize=(15, 11))
        plt.savefig(os.path.join(self.outdir, 'top_loss',
                                 self.model_name + '_' + str(datetime.datetime.now().strftime("%m%d%Y-%H%M"))))

    def save_hyperparameters(self,filename='hyperparameters'):
        file = open(
            os.path.join(self.outdir, 'hyperparameters',
                         filename + '_' + self.model_name + '_' + str(
                             datetime.datetime.now().strftime("%m%d%Y-%H%M")) + '.txt'), 'w')
        file.write(
            'hyper-parameters for model at {} \n'.format(
                str(datetime.datetime.now().strftime("%m%d%Y-%H%M"))))
        file.write('Resnet type is: {} \n'.format(self.model_dict[self.arch][0]))
        file.write('model name is: {} \n'.format(self.model_name))
        print('model name is: {} \n'.format(self.model_name))
        file.write('training name is: {} \n'.format(self.tr_name))
        file.write('validation name is: {} \n'.format(self.val_name))
        file.write('image size is: {} \n'.format(self.img_sz))
        file.write('learning rate for dense connected is: {} \n'.format(self.lr))
        print('learning rate for dense connected is: {} \n'.format(self.lr))
        file.write('learning rate range for whole network is: {} \n'.format(self.lr_range))
        print('learning rate for dense connected is: {} \n'.format(self.lr))
        file.write('batch size is: {} \n'.format(self.bs))
        file.write('this model was trained on device: {} \n'.format(self.device))
        print('this model was trained on device: {} \n'.format(self.device))
        file.write('number epochs densely connnected: {} \n'.format(self.dc_e))
        file.write('number epochs all layers: {} \n'.format(self.all_e))
        file.write('unfreeze?: {} \n'.format(str(self.unfreeze)))
        file.write('weighting: {} \n'.format(str(self.weightedloss)))
        file.write('weighting: {} \n'.format(str(self.weights)))
        file.close()

    def make_filestructure(self):
        '''
        make the file structure to write out all saved files
        :return:
        '''
        # make the filestructure for saving
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, 'confusion_matrix')):
            os.mkdir(os.path.join(self.outdir, 'confusion_matrix'))
        if not os.path.isdir(os.path.join(self.outdir, 'top_loss')):
            os.mkdir(os.path.join(self.outdir, 'top_loss'))
        if not os.path.isdir(os.path.join(self.outdir, 'saved_models')):
            os.mkdir(os.path.join(self.outdir, 'saved_models'))
        if not os.path.isdir(os.path.join(self.outdir, 'hyperparameters')):
            os.mkdir(os.path.join(self.outdir, 'hyperparameters'))
        if not os.path.isdir(os.path.join(self.outdir, 'exported_models')):
            os.mkdir(os.path.join(self.outdir, 'exported_models'))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε: float = 0.1, reduction='mean'):
        super().__init__()
        self.ε, self.reduction = ε, reduction

    def lin_comb(self,v1, v2, beta): return beta*v1 + (1-beta)*v2

    def reduce_loss(self,loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.lin_comb(loss / c, nll, self.ε)


if __name__ == '__main__':
    c = VisionMulticlass()
    name = c.train()