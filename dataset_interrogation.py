import os
import re
import numpy as np
import pandas as pd
#import pydicom
import shutil
import statistics
from collections import Counter
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
#import nibabel as nib

#author @t_sanf

class DatasetInterrogate:
    '''misc functions to help perform data analysis'''

    def __init__(self):
        self.basePATH=r'M:'
        self.database='MRIClinical'
        self.devfolder='consecutive'

    def check_correct_sorting(self):
        '''function to ensure check there is no training data in the validation or testing ddatasets'''

        unique_mrns={}
        for db in ('train','val','test'):
            unique_mrns[db]=set(file.split('_')[0] for file in get_slices(dir=db))  #get_slices --> helper function at bottom
            print('{} unique mrns in database {}'.format(len(unique_mrns[db]),db))

        print("mrns overlap between training and validation sets: {}".format(unique_mrns['train'].intersection(unique_mrns['val'])))
        print("mrns overlap between training and test sets: {}".format(unique_mrns['train'].intersection(unique_mrns['test'])))

    def check_counts_by_folder(self):
        '''function that evaluates how many unique patients, tumors, and slices are in the training, validation, and test sets'''

        base_path=os.path.join(self.basePATH,self.database,self.devfolder)
        all_slice_list=[]
        for db in ('train','val','test'):
            PIRADS_dict = {k: [] for k in ['2', '3', '4', '5']}
            print("-----------------------")
            print("database {}".format(db))
            slices=get_slices(path=base_path,dir=db)
            for slice in slices:
                all_slice_list+=[slice]
                for PIRADS in ['2','3','4','5']:
                    if slice.split('_')[8]==PIRADS:
                        PIRADS_dict[PIRADS].append(slice)
            for PIRADS in ['2','3','4','5']:
                files=PIRADS_dict[PIRADS]
                patients=[file.split('_')[0] for file in files]
                tumors = [file.split('_')[0]+'_'+file.split("_")[1]+'_'+file.split("_")[3] for file in files]

                print("for PIRADS {} lesions in dataset {}".format(PIRADS, db))
                print("Total slices: {}".format(len(files)))
                print("total unique tumors: {}".format(len(set(tumors))))
                print("total of {} unique patients".format(len(set(patients))))

    def check_PIRADS_counts_by_db(self):
        '''  '''
        base_path=os.path.join(self.basePATH,self.database,self.devfolder)
        PIRADS_list=[]
        PZ_TZ=[]
        lg_tz=[]
        nl=[]
        two=[]; three=[]; four=[]; five=[]
        segmentor=[]
        for pt in sorted(os.listdir(base_path)):
            for file in sorted(os.listdir(os.path.join(base_path,pt,'nifti','mask'))):
                if len(file.split('_'))>5:

                    PIRADS_list+=[file.split('_')[5]]
                    PZ_TZ+=[file.split('_')[3]]
                    if file.split('_')[5] == '5':
                        three
                        if file.split('_')[3]=='TZ':
                            lg_tz+=[pt]

                if len(file.split('_'))>2 and len(file.split('_'))<5:
                    if file.split('_')[0]=='WP':
                        if file.split('_')[2]=='1':
                            nl+=[pt]

                if file.split('_')[0]=='urethra':
                    segmentor+=[file.split('_')[1]]

        print("PIRADS scores in database {} are {}:".format(self.devfolder,Counter(PIRADS_list)))
        print("PZ TZ in database {}".format(Counter(PZ_TZ)))
        print("large tz lesions {}".format(len(set(lg_tz))))
        print("total of {} normal prostates".format(len(set(nl))))
        print(Counter(segmentor))

    def ID_label_errors(self,ext='voi'):
        '''recursively looks through filetree and finds files that have not been labeled according to our conventions'''
        problems=[]
        for root, dirnames, filenames in os.walk(self.basePATH):
            for filename in filenames:
                if filename.endswith(ext):
                    filename_list=filename.split('_')
                    if 'PIRADS' in filename_list and filename_list[len(filename_list)-1]=='bt.voi' and not filename_list[1]=='p':
                        if not filename_list[4]=='PIRADS':
                            problems+=[root+'_'+filename+'_'+' location issue']
        print(sorted(problems))

#################################################################################################################

class DatasetSummary:

    def __init__(self):
        self.basePATH=r'C:\Users\sanfordt\Desktop\PIRADS_dataset_updated'
        self.database='revision_analysis'
        self.workingdb='PIRADS_dataset_DO_NOT_TOUCH'
        self.devfolder='model_dev_indvPIRADS'

    def lesions_summary(self):
        '''in file called 'tumors' evaluates the dataset characteristics'''
        tumors=os.listdir(os.path.join(self.basePATH,self.database,'tumors')) #total number of tumors
        studies=pd.Series(file.split('_')[0] for file in os.listdir(os.path.join(self.basePATH,self.database,'lesions_by_study'))) #total number MRIs
        duplicated=studies[studies.duplicated()] #check to make sure there are no duplicates
        print("The following studies are duplicated {}".format(duplicated))
        num_unique_patients=len(list(set([file.split('_')[0] for file in os.listdir(os.path.join(self.basePATH,self.database,'tumors'))])))

        side=[]; location=[]; zone=[]; PIRADS=[];
        for tumor in tumors:
            side += [tumor.split("_")[3]]
            location += [tumor.split("_")[4]]
            zone += [tumor.split("_")[5]]
            PIRADS += [tumor.split("_")[7]]

        print("number tumors is is {}".format(len(tumors)))
        print("number of studies is {}".format(studies.count()))
        print("number of patients is {}".format(num_unique_patients))
        print("side of right sided tumors is  {}".format(Counter(side)))
        print('location of patients is {}'.format(Counter(location)))
        print('zone of patients is {}'.format(Counter(zone)))
        print('PIRADS score is {}'.format(Counter(PIRADS)))

    def demographic_data(self,db='consecutive'):
        '''
        obtaining demographics for each dataset individually.
        '''

        patients=os.listdir(os.path.join(self.basePATH,self.database,db)) #lesions with PIRADS 2 or greater
        tumors_included=os.listdir(os.path.join(self.basePATH,self.workingdb,'lesions_by_study'))  #change this part if you update your database, this is all patients included in the database
        excluded=set(patients).difference(set(tumors_included))  #patients that have PIRADS 1
        patients_overlap=list(set(patients).intersection(set(tumors_included))) #patient that have PIRADS 2 or greater
        mrns=pd.Series([patient.split('_')[0] for patient in patients_overlap])  #get all mrns in your database
        duplicated=mrns[mrns.duplicated()] #sanity check to make sure no repeat patients.

        #loop over patients, read in dicom files and extract demographic data
        age_list=[]; weight_list=[]; total=0
        for patient in patients_overlap:
            t2_path=os.path.join(self.basePATH,self.database,db,patient,'dicoms','t2')
            dcm=pydicom.dcmread(os.path.join(t2_path,os.listdir(t2_path)[0]))
            weight=dcm[0x00101030].value
            DOB = dcm[0x00100030].value
            DOB_datetime = date(year=int(DOB[0:4]), month=int(DOB[4:6]), day=int(DOB[6:8]))
            age = calculate_age(DOB_datetime)  #helper function
            age_list+=[age]
            weight_list+=[weight]
            total+=1

        weight_list=[val for val in weight_list if val>18]
        median_age=statistics.median(age_list)
        min_age=min(age_list); max_age=max(age_list)
        median_weight=statistics.median(weight_list)
        min_weight=min(weight_list); max_weight=max(weight_list)

        print("for database {}".format(db))
        print("total of {} studies".format(len(patients)))
        print("total of {} studies excluded".format(len(excluded)))
        print("The following studies are duplicated {}".format(duplicated))
        print("total of {} studies with >PIRADS 2".format(len(patients_overlap)))
        print("median age is {} with min age of {} and max age of {}".format(median_age,min_age,max_age))
        print("median weight is {} with min weight of {} and max weight of {}".format(median_weight,min_weight,max_weight))

    def ER_coil(self, db='consecutive'):
        '''this function returns a count of the highB value headers.
        At our instiution, b value of 2000 used only with endorectal coil.  All others are with b1500'''
        out_list = []
        patients = os.listdir(os.path.join(self.basePATH, self.database, db))
        tumors_included = os.listdir(os.path.join(self.basePATH, self.workingdb,'lesions_by_study'))  # change this part if you update your database
        patients_overlap = list(set(patients).intersection(set(tumors_included)))
        for patient in patients_overlap:
            highbs = os.listdir(os.path.join(self.basePATH,self.database,db,patient,'dicoms', 'highb','raw'))
            ds = pydicom.dcmread(os.path.join(self.basePATH,self.database,db,patient,'dicoms', 'highb','raw', highbs[0]))
            out_list += [ds[0x08, 0x103e].value]
        print(Counter(out_list))

    def calc_volumes(self,filetype='wp'):
        '''
        calculate the volumes for a segmented structure based on .nifti mask volume
        :return:
        '''

        #select patients with >PIRADS 2
        ds_tumors=os.path.join(self.basePATH,'databases',self.database)
        PIRADS_2_5=os.path.join(self.basePATH,'lesions_by_database',self.database+'_lesions','tumors')
        patients=overlap_pts(ds_tumors,PIRADS_2_5)

        outDF=pd.DataFrame()
        for patient in patients:
            # calculate voxel size size
            first_t2=os.listdir(os.path.join(self.basePATH,'databases',self.database, patient, 'dicoms', 't2'))[0]
            ds=pydicom.dcmread(os.path.join(self.basePATH,'databases',self.database,patient,'dicoms','t2',first_t2))
            xy_size=ds[0x28,0x30].value; z_size=ds[0x18,0x88].value
            volume_voxel=xy_size[0]*xy_size[1]*z_size

            #search the files for the type you are interested in (i.e. wp=whole prostate)
            filelist = []
            for file in os.listdir(os.path.join(self.basePATH,'databases',self.database, patient,'nifti','mask')):
                if len(file.split('_'))<5:
                    if filetype == 'wp': pat = re.compile('([Ww][Pp]){1}')
                    elif filetype == 'tz': pat = re.compile('([Tt][Zz]){1}')
                    if re.search(pat,file) !=None: filelist+=[file]

            name=''
            #select annotations for expert
            for i in range(len(filelist)):
                name=filelist[i]
                name_noend=name.split('.nii')[0]
                if name_noend.split('_')[-1]=='bt':filename=name
                elif name_noend.split('_')[-1]=='mm':filename=name
                elif name_noend.split('_')[-1]=='ts':filename=name
                elif name_noend.split('_')[-1] == 'pseg': filename = name
                elif name_noend.split('_')[-1] == 'dk':filename = name

            #calculate volume
            nifti_path=os.path.join(self.basePATH,'databases',self.database,patient,'nifti','mask',name)
            volume=calculate_volume(nifti_path, volume_voxel)  #helper function listed below
            print("volume of {} for patient {} is: {}".format(filetype,patient.split('_')[0],volume))
            series = pd.DataFrame([patient, volume]).transpose()
            outDF = pd.concat([outDF, series], axis=0)

        #save all volumes to file
        outDF.to_csv(os.path.join(self.basePATH,self.database+'_volumes_of_'+filetype+'.csv'))


###########################################################################
####################### Helper Functions #################################
###########################################################################


def get_slices(path,dir='', ext='.voi'):
    '''recursively looks in folder for files with specific file extension'''
    num = 0
    list_filenames = []
    for root, dirnames, filenames in os.walk(os.path.join(path,dir)):
        for filename in filenames:
            if filename.endswith(ext):
                num += 1
                list_filenames += [filename]
    unique_pts=list(set(file.split('_')[0] for file in list_filenames))
    print("total of {} files in the directory {} ".format(len(list_filenames), dir))
    print("total of {} unique filenames in the directory {}".format(len(unique_pts),dir))
    return (list_filenames)


def calculate_volume(path, volume_voxel):
    ''' calculate the volume of a structure for one patients for one patient'''
    # calculate volume
    volume = nib.load(path)
    vol_array = volume.get_fdata()
    return round((int(vol_array.sum()) * volume_voxel) / 1000,2)

def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

def overlap_PIRADS(ds_tumors,path_PIRADS2_5):
    ''' selects only the patients with PIRADS >2'''
    ds_tumors=os.listdir(ds_tumors)
    PIRADS2_5=[file.split('_')[0]+'_'+file.split('_')[1] for file in os.listdir(path_PIRADS2_5)]
    return list(set(ds_tumors).intersection(set(PIRADS2_5)))  # patient that have PIRADS 2 or greater

def overlap_pts():
    '''checks for overlap between two files'''
    path1=r''
    path2=r''
    overlap=set([file.split('_')[0] for file in os.listdir(path1)]).intersection(set([file.split('_')[0] for file in os.listdir(path2)]))
    print(sorted(overlap))

def get_filenames() -> object:

    path=r'C:\Users\sanfordt\Desktop\PIRADS_dataset\lesions_by_database\prostateX_lesions\tumors'
    df=pd.DataFrame(os.listdir(path))
    df.to_csv(r'C:\Users\sanfordt\Desktop\PIRADS_dataset\revision_analysis\output\eval_model\prostateX_names.csv')


def find_missing_wp_tz():

    path=r'C:\Users\sanfordt\Desktop\PIRADS_dataset_DO_NOT_TOUCH\databases\prostateX'
    patients=os.listdir(path)
    wp=[]; tz=[]
    for patient in patients:
        segs=os.listdir(os.path.join(path,patient,'voi'))
        for seg in segs:
            if seg.endswith('.voi'):
                if seg.split('_')[0]=='wp':
                    wp+=[patient]
                if seg.split("_")[0]=='tz':
                    tz+=[patient]

def plot_cm():
    two=[2,6,10,2]
    three=[0,36,27,8]
    four=[0,13,56,13]
    five=[0,3,12,46]
    array=[two,three,four,five]
    df_cm=pd.DataFrame(array,index=['PIRADS 2','PIRADS 3','PIRADS 4','PIRADS 5'],columns=['Predicted PRIADS 2','Predicted PRIADS 3','Predicted PRIADS 4','Predicted PRIADS 5'])
    plt.figure(figsize=(10,7))
    plt.figure(df_cm,annot=True)
    plt.show()

def plot_cm_validation():
    two=[2,1,3,1]
    three=[2,8,8,2]
    four=[1,0,6,1]
    five=[1,1,4,9]
    array=[two,three,four,five]
    df_cm=pd.DataFrame(array,index=['PIRADS 2','PIRADS 3','PIRADS 4','PIRADS 5'],columns=['Predicted PRIADS 2','Predicted PRIADS 3','Predicted PRIADS 4','Predicted PRIADS 5'])
    plt.figure(figsize=(10,7))
    plt.figure(df_cm)
    plt.show()


def calculate_overlap():
    '''overlap'''
    data=pd.read_csv(r'C:\Users\sanfordt\Desktop\blinded_second reader_EBT.csv')
    PIRADS_dict={'1':[],'2':[],'3':[],'4':[],'5':[]}
    agreement=0; within_1=0; upgraded=0; downgraded=0
    for index in data.index:
        ET_PIRADS=int(data.loc[index,'ET_overall'])
        GT_PIRADS=int(data.loc[index, 'BT_overall'])
        if ET_PIRADS==1:
            ET_PIRADS=2
        if GT_PIRADS==1:
            GT_PIRADS=2
        PIRADS_dict[str(GT_PIRADS)].append(ET_PIRADS)
        if ET_PIRADS == GT_PIRADS:
            agreement+=1
        if abs(ET_PIRADS-GT_PIRADS)<2:
            within_1+=1
        if ET_PIRADS>GT_PIRADS:
            upgraded+=1
        if ET_PIRADS<GT_PIRADS:
            downgraded+=1

    print("Agreement {}, ({}%)".format(agreement,round(agreement/data.shape[0],2)))
    print("Within 1 {}, ({}%)".format(within_1, round(within_1 / data.shape[0], 2)))
    print("Upgraded {}, ({}%)".format(upgraded, round(upgraded / data.shape[0], 2)))
    print("Downgraded {}, ({}%)".format(downgraded, round(downgraded / data.shape[0], 2)))

    for key in PIRADS_dict.keys():
        print("For values associated with {}".format(key))
        print(pd.Series(PIRADS_dict[key]).value_counts())







if __name__=='__main__':
    c=DatasetInterrogate()
    c.check_PIRADS_counts_by_db()