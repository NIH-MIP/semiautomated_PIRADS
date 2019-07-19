import os
import sys
import datetime

import fastai.vision as faiv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from fastai.vision import *
from fastai.callbacks import *
from fastai.callbacks.tracker import *
from fastai.basic_train import *

#authors @t_sanf @drsharmon

class MultiClassProstate:

    def __init__(self):
        self.imagedir='(insert path)'
        self.outdir='(insert path)'
        self.testPath=os.path.join(self.imagedir,'test')
        self.model_name='train'
        self.tr_name='train'
        self.val_name='val'
        self.resnet='resnet50'
        self.img_sz=224
        self.lr=0.00001
        self.lr_range=slice(1e-7, 1e-4)
        self.bs=64 #batch size
        self.device=0 #which gpu
        self.dc_e=100 #last layers epochs
        self.all_e=20 #all epochs
        self.lighting=0.1
        self.rotate=45
        self.early_stopping=True
        self.unfreeze=False
        self.weights = [10.,4.,2.,1.]
        self.model_dict={'resnet18':[models.resnet18],'resnet34':[models.resnet34],'resnet50':[models.resnet50],
            'resnet101':[models.resnet101],'resnet152':[models.resnet152]}


        #retraining
        self.retrain=False
        self.retraindir = '(insert path)'
        self.save_model_name='(insert name)'

    def load_jpg_from_folder(self):
        '''
        expects a path to a base folder with multiple subfolders including 'training', 'testing', etc
        :param path:
        :return: databunch
        '''

        #tfms =([*rand_pad(padding=3, size=self.img_sz, mode='zeros')], [])
        # tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
        #tfms = get_transforms(flip_vert=True, max_rotate=self.rotate, max_lighting = self.lighting, max_zoom=1.05, zoom_crop(scale=(0.75,2), do_rand=True, p=1.0))
        #extralist = [*rand_resize_crop(size=224, max_scale=1.5)]
        #extralist = [crop_pad(size=448,row_pct=0.5,col_pct=0.5),]
        #tfms = get_transforms(flip_vert=True,max_rotate=45,max_warp=0.1,max_lighting = 0.3,p_lighting=0.75,p_affine=0.5)#, xtra_tfms=extralist)
        tfms = get_transforms(flip_vert=True, max_rotate=45, max_warp=0.1, max_lighting=0.1, p_lighting=0.5,p_affine=0.5)

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

        #train loop
        w = torch.cuda.FloatTensor(self.weights)
        if self.early_stopping==True:
            learn = cnn_learner(data, self.model_dict[self.resnet][0],
                            metrics=[error_rate],
                            callback_fns=[ShowGraph,
                                          partial(SaveModelCallback, monitor='val_loss', mode='auto', name=self.tr_name),
                                          partial(EarlyStoppingCallback, monitor='val_loss', min_delta=0.001, patience=200)],
                            wd=0.1,
                            loss_func=torch.nn.CrossEntropyLoss(weight=w))

        else:
            learn = cnn_learner(data, self.model_dict[self.resnet][0],
                            metrics=[error_rate],
                            wd=0.1,
                            loss_func=torch.nn.CrossEntropyLoss(weight=w))


        self.trainblock(learner=learn)
        save_name = self.model_name + "_" + str(datetime.datetime.now().strftime("%m%d%Y-%H%M"))

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
        interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
        plt.savefig(os.path.join(self.outdir, 'confusion_matrix',
                                 self.model_name + '_' + str(datetime.datetime.now().strftime("%m%d%Y-%H%M")+'.jpg')))
        interp.plot_top_losses(9, figsize=(15, 11))
        plt.savefig(os.path.join(self.outdir, 'top_loss',
                                 self.model_name + '_' + str(datetime.datetime.now().strftime("%m%d%Y-%H%M")+'.jpg')))

    def save_hyperparameters(self,filename='hyperparameters'):
        file = open(
            os.path.join(self.outdir, 'hyperparameters',
                         filename + '_' + self.model_name + '_' + str(
                             datetime.datetime.now().strftime("%m%d%Y-%H%M")) + '.txt'), 'w')
        file.write(
            'hyper-parameters for model at {} \n'.format(
                str(datetime.datetime.now().strftime("%m%d%Y-%H%M"))))
        file.write('Resnet type is: {} \n'.format(self.model_dict[self.resnet][0]))
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

    ####################
    ## Apply Functions##
    ####################

    def convert_model_to_export(self, model_name):
        initial_filename = os.path.join(self.outdir, 'exported_models', model_name)
        final_filename = os.path.join(self.outdir, 'exported_models', 'export.pkl')
        shutil.copy2(initial_filename, final_filename)

    def apply_test_vote_binary(self):
        torch.cuda.set_device(self.device)

        # set up the output directory
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, 'val_results')):
            os.mkdir(os.path.join(self.outdir, 'val_results'))

        model_path = os.path.join(self.outdir, 'exported_models')
        learn = load_learner(model_path)
        test_path = self.testPath

        total_patients = 0
        correct = 0
        gt_dict = {'2': 0, '3_4_5': 0}
        correct_dict = {'2': 0, '3_4_5': 0}

        df_out = pd.DataFrame()
        for patient in os.listdir(os.path.join(test_path)):
            print(patient)
            print('patient')
            total_patients += 1

            sum_pred = np.zeros(4)
            square_pred = np.zeros(4)
            img_num = 0

            for image in sorted(os.listdir(os.path.join(test_path, patient))):
                img = open_image(os.path.join(test_path, patient, image))
                pred_class, pred_idx, outputs = learn.predict(img)
                print(outputs.numpy())
                sum_pred += outputs.numpy()
                square_pred += (outputs.numpy()) ** 2
                img_num += 1

            # metrics
            average = sum_pred / img_num
            sum_pred_class = np.argmax(sum_pred)
            ave_pred_class = np.argmax(average)
            square_pred_class = np.argmax(square_pred)

            print('sum prediction {}'.format(sum_pred))
            print('average prediction {}'.format(average))
            print('square prediction {}'.format(square_pred))

            # change this part to change the method of classification
            pred_PIRADS = str(ave_pred_class + 2)

            # check if prediction agrees with the ground truth
            gt = image.split('_')[8]

            print('Predicted PIRADS is {}'.format(pred_PIRADS))
            print('Ground Truth PIRADS is {}'.format(gt))
            gt_dict[str(gt)] += 1
            if pred_PIRADS == gt:
                print("correct")
                correct += 1
                correct_dict[str(gt)] += 1

            if pred_PIRADS != gt:
                print("incorrect")
                t_df = pd.DataFrame([image, pred_PIRADS, gt]).transpose()
                df_out = pd.concat([df_out, t_df], axis=0)

        print("------------------------------")
        print("total patients {}".format(total_patients))
        print("total correct {}".format(correct))
        print("percent_correct {}".format(correct / total_patients))
        print(gt_dict)
        print(correct_dict)
        for i in range(2, 5):
            print('percent correct for PIRADS {} is {}'.format(i, correct[str(i)] / gt_dict[str(i)]))

        # write dataframe out to csv
        df_out.columns = ['img_name', 'class_pred', 'ground truth', ]
        df_out.to_csv(os.path.join(self.outdir, 'val_results', 'val_results_by_img' + '_' + self.model_name + '_' + str(
            datetime.datetime.now().strftime("%m%d%Y-%H%M")) + '.csv'))


    def apply_test_vote_multiclass(self):
        '''
        applies model on patient level
        '''

        torch.cuda.set_device(self.device)

        # set up the output directory
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, 'val_results')):
            os.mkdir(os.path.join(self.outdir, 'val_results'))

        model_path = os.path.join(self.outdir, 'exported_models')
        learn = load_learner(model_path)
        test_path = self.testPath

        total_patients = 0
        correct = 0
        gt_dict={'2':0,'3':0,'4':0,'5':0}
        correct_dict={'2':0,'3':0,'4':0,'5':0}

        df_out = pd.DataFrame()
        for patient in os.listdir(os.path.join(test_path)):
            print(patient)
            print('patient')
            total_patients+=1

            sum_pred=np.zeros(4)
            square_pred=np.zeros(4)
            img_num=0

            for image in sorted(os.listdir(os.path.join(test_path, patient))):
                img = open_image(os.path.join(test_path, patient, image))
                pred_class, pred_idx, outputs = learn.predict(img)
                print(outputs.numpy())
                sum_pred+=outputs.numpy()
                square_pred+=(outputs.numpy())**2
                img_num+=1

            #metrics
            average=sum_pred/img_num
            sum_pred_class=np.argmax(sum_pred)
            ave_pred_class=np.argmax(average)
            square_pred_class=np.argmax(square_pred)

            print('sum prediction {}'.format(sum_pred))
            print('average prediction {}'.format(average))
            print('square prediction {}'.format(square_pred))

            #change this part to change the method of classification
            pred_PIRADS=str(ave_pred_class+2)

            #check if prediction agrees with the ground truth
            gt=image.split('_')[8]

            print('Predicted PIRADS is {}'.format(pred_PIRADS))
            print('Ground Truth PIRADS is {}'.format(gt))
            gt_dict[str(gt)]+=1
            if pred_PIRADS==gt:
                print("correct")
                correct+=1
                correct_dict[str(gt)]+=1

            if pred_PIRADS!=gt:
                print("incorrect")
                t_df = pd.DataFrame([image, pred_PIRADS, gt]).transpose()
                df_out = pd.concat([df_out, t_df], axis=0)

        print("------------------------------")
        print("total patients {}".format(total_patients))
        print("total correct {}".format(correct))
        print("percent_correct {}".format(correct/total_patients))
        print(gt_dict)
        print(correct_dict)
        for i in range(2,5):
            print('percent correct for PIRADS {} is {}'.format(i,correct[str(i)]/gt_dict[str(i)]))


        # write dataframe out to csv
        df_out.columns = ['img_name', 'class_pred', 'ground truth', ]
        df_out.to_csv(os.path.join(self.outdir, 'val_results', 'val_results_by_img' + '_' + self.model_name + '_' + str(
            datetime.datetime.now().strftime("%m%d%Y-%H%M")) + '.csv'))



if __name__ == '__main__':
    c = MultiClassProstate()
    name = c.train()
    c.convert_model_to_export(model_name=name)
    c.apply_test_vote_multiclass()
