import os
import pandas as pd
import shutil
import random
import re
from distutils.dir_util import copy_tree

class DevelopDataset:
    '''class to split the data into train/val/test datasets - need to run segment_save_jpg first
    note - the only folders needed are lesion_by_study, T2, tumors by annotator
    if you re-process, delete tumors_by_pt, model_dev_indvPIRADS, tumors

    '''
    random.seed(10)

    def __init__(self):
        self.basePATH=r'C:\Users\sanfordt\Desktop\PIRADS_dataset_updated\revision_analysis'
        self.lesions=os.path.join(self.basePATH,'lesions_by_study')
        self.tumors_by_pt=os.path.join(self.basePATH,'tumors_by_pt')
        self.tumors = os.path.join(self.basePATH,'tumors')
        self.model_dev=os.path.join(self.basePATH,'model_dev_indvPIRADS_1')

    def process(self,need_preprocess=True):
        '''
        note2 --> to redo the dataset needed (i.e. diff padding), set pre-process to True
        note1 --> if you want to rerun, need to delete the jpg,tumors_by_pt,tumors,model_dev folders
        '''

        #set up the file structure
        if not os.path.exists(os.path.join(self.tumors_by_pt)):
            os.mkdir(os.path.join(self.tumors_by_pt))
        if not os.path.exists(os.path.join(self.tumors)):
            os.mkdir(os.path.join(self.tumors))

        # getting all tumors into one folder called 'all tumors'
        if need_preprocess==True:
            self.comb_slices_by_tumor()
            self.sort_tumors()
            self.split_train_val_test()
            #self.anonymize() --> need to fix if use again

        # use this verson if you have already created 'all_tumors" and just need to remake model_dev folder
        if need_preprocess==False:
            self.split_train_val_test()
            # self.anonymize() --> need to fix if use again


    def split_train_val_test(self,val_fraction=0.222,test_fraction=0.1,balance=False,clip=False):
        '''split data into train/val and test sets, with test set by by patient and train/val by image
        :param val_factor - percent to be left out as test set on PATIENT level
        :param test_faction - percent to be left out as test set on PATIENT level
        :param: balance (bool) --> perform oversampling of the minority class
        '''

        #first, get a list of all the patients
        patients=[patient for patient in os.listdir(self.tumors_by_pt)]

        #make test set on patient level, subtract these patients from others
        test_patients=[patients[i] for i in random.sample(range(len(patients)), int(len(patients) * test_fraction))]
        train_val_patients=list(set(patients)-set(test_patients))
        val_patients=[train_val_patients[i] for i in random.sample(range(len(train_val_patients)), int(len(train_val_patients) * val_fraction))]
        train_patients = list(set(train_val_patients) - set(val_patients))

        print('total of {} test patients'.format(len(test_patients)))
        print('total of {} val patients'.format(len(val_patients)))
        print('total of {} train patients'.format(len(train_patients)))

        #obtain test_tumors
        test_sample=self.find_sample(test_patients); val_sample=self.find_sample(val_patients); train_sample=self.find_sample(train_patients)
        train_val_dict={'train':train_sample,'val':val_sample}

        #set up basic file structure:
        #set up file structure
        if not os.path.exists(self.model_dev):
            os.mkdir(self.model_dev)
        for filetype in ['train','val','test','val_pt','all_train_images','all_val_images']:
            if not os.path.exists(os.path.join(self.model_dev, filetype)):
                os.mkdir(os.path.join(self.model_dev, filetype))
        for filetype in ['train','val']:
            for PIRADS in ['PIRADS_2','PIRADS_3','PIRADS_4','PIRADS_5']:
                if not os.path.exists(os.path.join(self.model_dev, filetype,PIRADS)):
                    os.mkdir(os.path.join(self.model_dev, filetype,PIRADS))

        #first, copy all tumor-level information into a new file
        print("making test set")
        for tumor in test_sample:
            print(tumor)
            copy_tree(os.path.join(self.tumors,tumor),os.path.join(self.model_dev,'test',tumor))

        print('making val_pt set')
        for tumor in val_sample:
            copy_tree(os.path.join(self.tumors, tumor),os.path.join(self.model_dev, 'val_pt', tumor))


        #next, copy all training images to an 'all folder'
        print('copying all train/val images to new folder')
        for dataset in train_val_dict.keys():
            print(dataset)
            sample=train_val_dict[dataset]
            for tumor in sample:
                files=os.listdir(os.path.join(self.tumors,tumor))
                if dataset == 'train':

                    if clip==True:
                        print("length of files is {} for tumor {} before clipping".format(len(files), tumor))
                        if len(files)>3:
                            del files[0]
                            del files[-1]
                            print("length of files is {} for tumor {} after clipping".format(len(files), tumor))

                for file in files:
                    shutil.copy2(os.path.join(self.tumors,tumor,file),\
                                 os.path.join(self.model_dev,'all_'+dataset+'_images',file))

        #split data by file
        for key in train_val_dict.keys():
            print("making {} set".format(key))
            for file in os.listdir(os.path.join(self.model_dev,'all_'+key+'_images')):
                if file != 'Thumbs.db':
                    print(file)
                    if file.split('_')[8]=='2':
                        shutil.copy2(os.path.join(self.model_dev,'all_'+key+'_images',file),
                                     os.path.join(self.model_dev,key,'PIRADS_2',file))
                    if file.split('_')[8]=='3':
                        shutil.copy2(os.path.join(self.model_dev,'all_'+key+'_images',file),
                                     os.path.join(self.model_dev,key,'PIRADS_3',file))
                    if file.split('_')[8]=='4':
                        shutil.copy2(os.path.join(self.model_dev,'all_'+key+'_images',file),
                                     os.path.join(self.model_dev,key,'PIRADS_4',file))
                    if file.split('_')[8]=='5':
                        shutil.copy2(os.path.join(self.model_dev,'all_'+key+'_images',file),
                                     os.path.join(self.model_dev,key,'PIRADS_5',file))
        if balance==True:
            self.balance_dataset(cat1='PIRADS_2',cat2='PIRADS_3',cat3='PIRADS_4',cat4='PIRADS_5')

    def find_sample(self,pts):
        '''helper function for listing tumors within specified patient directory'''
        pts_out=[]
        for pat in pts:
            pts_out+=[pat+'_'+tumor for tumor in os.listdir(os.path.join(self.tumors_by_pt, pat))]
        return pts_out

    def get_tumors(self,list_tumors):
        '''helper function to find tumor slices within list of tumor names'''
        slices=[]
        for tumor in list_tumors:
            slices+=[os.path.join(self.tumors,slice) for slice in os.listdir(os.path.join(self.tumors,tumor))]
        return slices

###############################
    ### Setting up data - getting all tumors grouped into folder called 'all_tumors'

    def sort_tumors(self):
        '''after running comb_slices_by_tumor below, this function will move all tumors to a new folder and move
        all slices to a new folder called 'tumors_all' and another one called 'tumors"

        '''

        for patient in os.listdir(os.path.join(self.tumors_by_pt)):
            for tumor_name in os.listdir(os.path.join(self.tumors_by_pt,patient)):
                if not os.path.exists(os.path.join(self.tumors, patient+'_'+tumor_name)):
                    os.mkdir(os.path.join(self.tumors, patient+'_'+tumor_name))
                    copy_tree(os.path.join(self.tumors_by_pt,patient,tumor_name),os.path.join(self.tumors, patient+'_'+tumor_name))


    def comb_slices_by_tumor(self,mid=False):
        '''uses dictionary generated by self.gen_dict_by_tumor() and copies all the files into folders by tumor
        to a folder called tumors_by_pt
        '''

        for patient_dir in os.listdir(os.path.join(self.lesions)):

            print("sorting tumors for patient {}".format(patient_dir))
            patient_dict=self.gen_dict_by_tumor(patient_dir)
            for key in patient_dict.keys():
                slices=patient_dict[key]

                if mid == True:
                    if len(slices)> 3:
                        pass
                jpg_name=slices[0].split('_')[3]+'_'+slices[0].split('_')[4]+'_'+slices[0].split('_')[5]+'_'+slices[0].split('_')[6]+'_'+slices[0].split('_')[7]+'_'+slices[0].split('_')[8]
                tumor_name=jpg_name.split('.')[0]

                patient_dir=slices[0].split('_')[0]+'_'+slices[0].split('_')[1]
                for slice in slices:
                    #make directory if one does not exist
                    if not os.path.exists(os.path.join(self.tumors_by_pt,patient_dir)):
                        os.mkdir(os.path.join(self.tumors_by_pt,patient_dir))
                    if not os.path.exists(os.path.join(self.tumors_by_pt,patient_dir,tumor_name)):
                        os.mkdir(os.path.join(self.tumors_by_pt, patient_dir,tumor_name))

                    initial_path=os.path.join(self.lesions,patient_dir,slice)
                    transfer_path=os.path.join(self.tumors_by_pt,patient_dir,tumor_name,slice)

                    shutil.copyfile(initial_path,transfer_path)


    def gen_dict_by_tumor(self,patient_dir='PEx0148_00000000'):
        '''helper function that puts all slices output by segment_save_jpg into a dictionary'''

        tumor_slices=[]
        nums=[]
        for file in os.listdir(os.path.join(self.lesions,patient_dir)):
            print(file)
            if file !='Thumbs.db':
                nums+=file.split('_')[3]
                tumor_slices+=[file]
        num_unique=list(set(nums))

        tumor_dict={}
        for num in num_unique:
            tumor_slices_by_tumor = []
            for tumor_slice in tumor_slices:
                if tumor_slice.split('_')[3]==num:
                    tumor_slices_by_tumor+=[tumor_slice]
                tumor_dict[num]=tumor_slices_by_tumor

        return(tumor_dict)

    def balance_dataset(self,cat1,cat2,cat3,cat4,type='all'):
        '''
        Figures out which subset of data is the smallest and oversamples until data are balanced
        :return:
        '''

        cat1_num = len(os.listdir(os.path.join(self.model_dev+'_'+type,'train',cat1)))
        cat2_num = len(os.listdir(os.path.join(self.model_dev+'_'+type, 'train', cat2)))
        cat3_num = len(os.listdir(os.path.join(self.model_dev + '_' + type, 'train', cat3)))
        cat4_num = len(os.listdir(os.path.join(self.model_dev + '_' + type, 'train', cat4)))

        cat_dict={cat1_num:cat1,cat2_num:cat2,cat3_num:cat3,cat4_num:cat4}
        largest_num=max(cat_dict.keys())
        cat_largest = cat_dict[largest_num]

        for cat_num in cat_dict.keys():
            if cat_num==cat_largest:
                continue
            else:
                cat1_num = largest_num
                cat1=cat_dict[largest_num]
                cat2_num=cat_num
                cat2=cat_dict[cat2_num]

            if cat1_num==cat2_num:
                break
            elif cat1_num>cat2_num:
                smaller=cat2; larger_num=cat1_num;smaller_num=cat2_num
            else:
                smaller=cat1; larger_num=cat2_num;smaller_num=cat1_num


            diff=larger_num-smaller_num
            print("difference between {} and {} classes is {}".format(cat1,cat2,diff))
            output=[os.path.join(self.model_dev+'_'+type,'train',smaller,file) for file in os.listdir(os.path.join(self.model_dev+'_'+type,'train',smaller))]
            output_series=pd.Series(output)
            val = output_series.sample(diff,replace=True)
            print('performing oversampling!')

            for file in val:
                jpeg_removed = file.split('.jpeg')[0]
                random_num=str(''.join(random.sample('0123456789', 5)))
                new_filename=jpeg_removed+random_num+'.jpeg'
                shutil.copy2(os.path.join(self.model_dev+'_'+type,'train',file),os.path.join(self.model_dev+'_'+type,'train',new_filename))

    def anon_recurs(self,path,extension='.npy'):
        '''
        recursively looks for all files with a given extension and anonymizes with 10 digit random number
        :return:
        '''
        list_extensions=['all_train_images','all_val_images','test','train','val','val_pt','PIRADS_2','PIRADS_3','PIRADS_4','PIRADS_5']
        files_and_images=os.listdir(path)
        for item in files_and_images:
            if item.endswith(extension):
                os.rename(os.path.join(path,item),os.path.join(path,str(random.randint(0000000000,9999999999))+extension))
            elif os.path.isdir(os.path.join(path,item)):
                if item in list_extensions:
                    os.chdir(os.path.join(path,item))
                    self.anon_recurs(path=os.path.join(path,item),extension=extension)
                elif item not in list_extensions:
                    new_item_name=str(os.path.join(path,str(random.randint(0000000000,9999999999))+extension))
                    os.rename(os.path.join(path,item),os.path.join(path,new_item_name))
                    os.chdir(os.path.join(path,new_item_name))
                    self.anon_recurs(path=os.path.join(path,new_item_name),extension=extension)


if __name__=='__main__':
    c=DevelopDataset()
    c.process()

