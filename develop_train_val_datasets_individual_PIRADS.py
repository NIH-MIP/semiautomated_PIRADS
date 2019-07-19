import os
import pandas as pd
import shutil
import random
import re
from distutils.dir_util import copy_tree

#author @t_sanf

class DevelopDataset:
    '''class to save MRI images in jpgs'''

    def __init__(self):
        self.jpg=r'(insert path)'
        self.jpg_tumors=r'(insert path)'
        self.tumors = r'(insert path)'
        self.model_dev=r'(insert path)'
        self.anonymous=r'(insert path)'


    def process(self,need_preprocess=False,type='all'):
        '''
        note2 --> to redo the dataset needed (i.e. diff padding), set pre-process to True
        note1 --> if you want to rerun, need to delete the jpg,jpg_tumors,tumors,model_dev folders
        '''
        random.seed(9001)

        #set up the file structure
        if not os.path.exists(os.path.join(self.jpg)):
            os.mkdir(os.path.join(self.jpg))
        if not os.path.exists(os.path.join(self.jpg_tumors)):
            os.mkdir(os.path.join(self.jpg_tumors))

        # getting all tumors into one folder called 'all tumors'
        if need_preprocess==True:
            self.comb_slices_by_tumor()
            self.sort_tumors(type=type)
            self.split_train_val_test(type=type)
            #self.anonymize() --> need to fix if use again

        # use this verson if you have already created 'all_tumors" and just need to remake model_dev folder
        if need_preprocess==False:
            self.split_train_val_test(type=type)
            #self.anonymize() --> unhash if you need data anonymized (to run on cluster) 


    def anonymize(self):
        random.seed(9001)
        out_dict={}

        for state in ['train','val']:
            for PIRADS in ['PIRADS_2_3','PIRADS_4_5']:
                for file in os.listdir(os.path.join(self.model_dev,state,PIRADS)):
                    new_name=random.randint(1000000000,9999999999)
                    out_dict[new_name]=file
                    shutil.copy2(os.path.join(self.model_dev,state,PIRADS,file),\
                                 os.path.join(self.anonymous,state,PIRADS,str(new_name)+'.jpg'))

        out_dict_df=pd.DataFrame.from_dict(out_dict)
        out_dict_df.to_csv(os.path.join(self.model_dev,'db_key','db_key.csv'))

    #def split_train_val(self,val_fraction=0.2,type='all'):



    def split_train_val_test(self,val_fraction=0.2,test_fraction=0.1,type='all',balance=False,clip=False):
        '''split data into train/val and test sets, with test set by by patient and train/val by image
        :param test_faction - percent to be left out as test set on TUMOR level
        :param test_faction - percent to be left out as test set on PATIENT level
        :param type - 'all' or 'pz' or 'tz
        :param: balance (bool) --> perform oversampling of the minority class
        '''

        #first, split data by patient into training and test
        tumors=[tumor for tumor in os.listdir(self.tumors+'_'+type)]
        test_sample=[tumors[i] for i in random.sample(range(len(tumors)),int(len(tumors)*test_fraction))]
        train_val_sample=list(set(tumors)-set(test_sample))
        val_sample = [train_val_sample[i] for i in random.sample(range(len(train_val_sample)), int(len(train_val_sample) * val_fraction))]
        train_sample = list(set(train_val_sample) - set(val_sample))
        train_val_dict={'train':train_sample,'val':val_sample}

        #set up basic file structure:
        #set up file structure
        if not os.path.exists(self.model_dev+'_'+type):
            os.mkdir(self.model_dev+'_'+type)
        for filetype in ['train','val','test','val_pt','all_train_images','all_val_images']:
            if not os.path.exists(os.path.join(self.model_dev + '_' + type, filetype)):
                os.mkdir(os.path.join(self.model_dev + '_' + type, filetype))
        for filetype in ['train','val']:
            for PIRADS in ['PIRADS_2','PIRADS_3','PIRADS_4','PIRADS_5']:
                if not os.path.exists(os.path.join(self.model_dev + '_' + type, filetype,PIRADS)):
                    os.mkdir(os.path.join(self.model_dev + '_' + type, filetype,PIRADS))

        #first, copy all tumor-level information into a new file
        print("making test set")
        for tumor in test_sample:
            copy_tree(os.path.join(self.tumors+'_'+type,tumor),os.path.join(self.model_dev+'_'+type,'test',tumor))

        print('making val_pt set')
        for tumor in val_sample:
            copy_tree(os.path.join(self.tumors + '_' + type, tumor),os.path.join(self.model_dev + '_' + type, 'val_pt', tumor))


        #next, copy all training images to an 'all folder'
        print('copying all train/val images to new folder')
        for dataset in train_val_dict.keys():
            print(dataset)
            sample=train_val_dict[dataset]
            for tumor in sample:
                files=os.listdir(os.path.join(self.tumors+'_'+type,tumor))
                if dataset == 'train':

                    if clip==True:
                        print("length of files is {} for tumor {} before clipping".format(len(files), tumor))
                        if len(files)>3:
                            del files[0]
                            del files[-1]
                            print("length of files is {} for tumor {} after clipping".format(len(files), tumor))

                for file in files:
                    shutil.copy2(os.path.join(self.tumors+'_'+type,tumor,file),\
                                 os.path.join(self.model_dev+'_'+type,'all_'+dataset+'_images',file))

        #split data by file
        for key in train_val_dict.keys():
            print("making {} set".format(key))
            for file in os.listdir(os.path.join(self.model_dev+'_'+type,'all_'+key+'_images')):
                if file != 'Thumbs.db':
                    print(file)
                    if file.split('_')[8]=='2':
                        shutil.copy2(os.path.join(self.model_dev+'_'+type,'all_'+key+'_images',file),
                                     os.path.join(self.model_dev+'_'+type,key,'PIRADS_2',file))
                    if file.split('_')[8]=='3':
                        shutil.copy2(os.path.join(self.model_dev+'_'+type,'all_'+key+'_images',file),
                                     os.path.join(self.model_dev+'_'+type,key,'PIRADS_3',file))
                    if file.split('_')[8]=='4':
                        shutil.copy2(os.path.join(self.model_dev+'_'+type,'all_'+key+'_images',file),
                                     os.path.join(self.model_dev+'_'+type,key,'PIRADS_4',file))
                    if file.split('_')[8]=='5':
                        shutil.copy2(os.path.join(self.model_dev+'_'+type,'all_'+key+'_images',file),
                                     os.path.join(self.model_dev+'_'+type,key,'PIRADS_5',file))
        if balance==True:
            self.balance_dataset(cat1='PIRADS_2',cat2='PIRADS_3',cat3='PIRADS_4',cat4='PIRADS_5',type=type)


###############################
    ### Setting up data - getting all tumors grouped into folder called 'all_tumors'


    def sort_tumors(self, type='all'):
        '''after running comb_slices_by_tumor below, this function will move all tumors to a new folder and move
        all slices to a new folder called 'tumors_all' and another one called 'tumors"
        :param type - either 'all' or 'pz' or 'tz

        '''

        for patient in os.listdir(os.path.join(self.jpg_tumors)):
            print(patient)

            #all tummors
            if type=='all':
                for tumor_name in os.listdir(os.path.join(self.jpg_tumors,patient)):

                    if not os.path.exists(os.path.join(self.tumors+'_'+type, patient+'_'+tumor_name)):
                        os.mkdir(os.path.join(self.tumors+'_'+type, patient+'_'+tumor_name))

                        copy_tree(os.path.join(self.jpg_tumors,patient,tumor_name),\
                                os.path.join(self.tumors+'_'+type, patient+'_'+tumor_name))

            elif type=='pz' or type=='tz':
                for tumor_name in os.listdir(os.path.join(self.jpg_tumors, patient)):
                    if type=='tz':
                        pat = re.compile('([Tt][Zz]){1}')
                    if type=='pz':
                        pat = re.compile('([Pp][Zz]){1}')

                    if pat.search(tumor_name) != None:

                        if not os.path.exists(os.path.join(self.tumors + '_' + type)):
                            os.mkdir(os.path.join(self.tumors + '_' + type))

                        if not os.path.exists(os.path.join(self.tumors + '_' + type, patient + '_' + tumor_name)):
                            os.mkdir(os.path.join(self.tumors + '_' + type, patient + '_' + tumor_name))

                            copy_tree(os.path.join(self.jpg_tumors, patient, tumor_name), \
                                      os.path.join(self.tumors+'_'+type, patient + '_' + tumor_name))

    def comb_slices_by_tumor(self,mid=True):
        '''uses dictionary generated by self.gen_dict_by_tumor() and copies all the files into folders by tumor
        to a folder called jpg_tumors
        '''

        for patient_dir in os.listdir(os.path.join(self.jpg)):

            print("sorting tumors for patient {}".format(patient_dir))
            patient_dict=self.gen_dict_by_tumor(patient_dir)
            for key in patient_dict.keys():
                slices=patient_dict[key]

                if mid == True:
                    if len(slices)> 3:
                        pass

                tumor_name=slices[0].split('_')[3]+'_'+slices[0].split('_')[4]+'_'+slices[0].split('_')[5]\
                           +'_'+slices[0].split('_')[6]+'_'+slices[0].split('_')[7]+'_'+slices[0].split('_')[8]
                tumor_name=tumor_name.split('.')[0]

                patient_dir=slices[0].split('_')[0]+'_'+slices[0].split('_')[1]
                for slice in slices:

                    #make directory if one does not exist
                    if not os.path.exists(os.path.join(self.jpg_tumors,patient_dir)):
                        os.mkdir(os.path.join(self.jpg_tumors,patient_dir))
                    if not os.path.exists(os.path.join(self.jpg_tumors,patient_dir,tumor_name)):
                        os.mkdir(os.path.join(self.jpg_tumors, patient_dir,tumor_name))

                    initial_path=os.path.join(self.jpg,patient_dir,slice)
                    transfer_path=os.path.join(self.jpg_tumors,patient_dir,tumor_name,slice)

                    shutil.copyfile(initial_path,transfer_path)


    def gen_dict_by_tumor(self,patient_dir='PEx0148_00000000'):
        '''helper function that puts all slices output by segment_save_jpg into a dictionary'''

        tumor_slices=[]
        nums=[]
        for file in os.listdir(os.path.join(self.jpg,patient_dir)):
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


if __name__=='__main__':
    c=DevelopDataset()
    c.process()

