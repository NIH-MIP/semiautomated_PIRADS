import os
import numpy as np
import pandas as pd
import pydicom
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import imageio
import scipy.misc

#author @t_sanf

#from Align import Align
from parsing_VOI import ParseVOI


class SegmentAligned(ParseVOI):

    def __init__(self):
        self.basePATH = r'(insert path)'
        self.databases=['surgery_cases','prostateX','consecutive']

        self.save=r'(insert path)'
        self.save_tumors=r'(insert path)'


    def segment_aligned_by_annotator(self,initials='bt'):
        '''
        creates bounding boxes based on segmentation in each image and saves as .jpg.
        ***note, need to delete the files in 'tumors by annotator' if you updated your dataset
        **note2 --> need to update development database if more tumors were annotated
        :return:
        '''

        self.find_tumors_by_annotator(initials='bt') #runs code and saves to disk to save time, need to delete file if want to update
        processed_files=pd.read_csv(os.path.join(self.save_tumors,initials+'_annotated_tumor_paths.csv'))
        processed_files=processed_files['filepaths'].tolist()
        print(len(processed_files))
        print(processed_files)

        #need to check if files already segmented
        files_already_done=os.listdir(os.path.join(self.save,'save_raw'))
        print("total of {} files already segmented".format(len(files_already_done)))
        files_to_segment=[]
        for file in processed_files:
            print(file)
            if str(file) not in files_already_done:
                files_to_segment+=[file]
        print("total of {} files to segment".format(len(files_to_segment)))

        fails=[]
        segmentation_fails=[]
        for file in files_to_segment:

            try:
                patient_dir=file.split("\\")[-3]
                database=file.split("\\")[-4]
                print(database)
                print("starting segmentation for patient {}".format(patient_dir))
                self.segment_aligned(database=database,patient_dir=patient_dir,voi_path=file)

            except:
                fails+=[patient_dir]
                print("patient_dir failed {}".format(file.split("\\")[-3]))

        print(fails)
        #segmentation_fails=pd.Series(segmentation_fails)
        #segmentation_fails.to_csv(os.path.join(self.failPATH,'segmentation_fails.csv'))


    def segment_aligned(self,database,patient_dir,voi_path):
        '''
        for files that are already aligned, this looks up dicom paths, voi paths, and segments all the images,
        then saves the T2 and the combination of 3 separately
        :param patient_dir --> output of completed_files
        :return: all segmented files
        '''

        dicom_paths=self.dicom_paths(database,patient_dir)
        print(dicom_paths)
        bboxes = self.BBox_from_position(voi_path)

        #start by iterating over bounding boxes
        index = 0
        for bbox in bboxes.keys():
            vals=bboxes[bbox] #select values for each bounding box
            print("vals are {}".format(vals))

            #for each bounding box, select the appropriate slice and segment
            segmented_image_dict={}
            for series in dicom_paths:
                paths=dicom_paths[series]
                path_to_load=paths[int(bbox)]
                segmented_image=self.segment_image(path_to_load,vals,patient_dir,index,series)
                segmented_image_dict[series]=segmented_image

            #extract each sequance array and combine into numpy array
            t2=segmented_image_dict['t2']; adc=segmented_image_dict['adc']; highb=segmented_image_dict['highb']
            stacked_image=np.dstack((t2,adc,highb))

            #normalize --> note, data is normalized based on slice level
            stacked_image[:,:,0]=self.rescale_array(stacked_image[:,:,0])
            stacked_image[:, :, 1] = self.rescale_array(stacked_image[:, :, 1])
            stacked_image[:, :, 2] = self.rescale_array(stacked_image[:, :, 2])

            #make a directory if one doesn't already exist and save npy files to it
            if not os.path.exists(os.path.join(self.save,'numpy', patient_dir)):
                os.mkdir(os.path.join(self.save,'numpy', patient_dir))
            os.chdir(os.path.join(self.save,'numpy', patient_dir))

            np.save(patient_dir + '_' + str(index) + '_' + vals[0] + '.npy',stacked_image)

            #make a directory if one doesn't already exist for images, conver to Image and save .jpg
            if not os.path.exists(os.path.join(self.save,'jpg', patient_dir)):
                os.mkdir(os.path.join(self.save,'jpg', patient_dir))
            os.chdir(os.path.join(self.save,'jpg', patient_dir))

            #opencv solution
            cv2.imwrite(patient_dir + '_' + str(index) + '_' + vals[0] + '.jpg',stacked_image)

            index+=1

    def dicom_paths(self, database,patient_dir):
        '''
        start in patient directory, then find all dicom files listed in that directory -->  note adc and highb
        are aligned to t2
        :param patient_dir (str): the folder name of the files
        :return: list of paths to dicom files within patient directory
        '''

        #set base directory
        dir=os.path.join(self.basePATH,database,patient_dir,'dicoms')

        #set path to each directory for aligned files
        t2_dir=os.path.join(dir,'t2')
        adc_dir=os.path.join(dir,'adc','aligned')
        highb_dir=os.path.join(dir,'highb','aligned')

        #setup dict names
        series_dict={'t2':t2_dir,'adc':adc_dir,'highb':highb_dir}

        #loop over series and joint with series dir path to get paths to dicom files for all images in all series
        dicom_dict={}
        for series in series_dict.keys():
            try:
                files=os.listdir(series_dict[series])
                path_list=[]
                for file in files:
                    path_list+=[os.path.join(series_dict[series],file)]
                path_list=self.order_dicom(path_list)
                dicom_dict[series]=path_list
            except:
                print("series {} not able to be loaded".format(series))
        return dicom_dict


    def voi_path(self, database,patient_dir,filetype='PIRADS'):
        '''
        function to find all .voi files within a specific patient directory.  Meant to be used within a for loop
        searching over all the files
        :param patient_dir (str) the filename of the patient
        :param type (str): type of segementation to perform
               PIRADS - tumor
               wp- whole prostate
               u -urethra
        :return: list of files that have the search term of interest (i.e. 'PIRADS')
        '''
        dir=os.path.join(self.basePATH,database, patient_dir, 'voi')
        files = os.listdir(dir)
        file_path=None
        for file in files:
            split_file=file.replace(' ','_').split('_')
            if filetype not in split_file:
                continue
            elif filetype in split_file:
                file_path=str(os.path.join(dir,file))
        return file_path

    def segment_image(self,path_to_image,vals,patient_dir,index,series,pad=10):
        '''
        helper function that takes in path to image, values and performs the segmentation
        :param path_to_image: self explanatory
        :param vals: indexes
        :param patient_dir:
        :param index:
        :param series:
        :param pad: number of voxes around the image in question
        :return:
        '''
        ds = pydicom.dcmread(path_to_image)
        data = ds.pixel_array
        data=self.rescale_array(data)  #normalize by slice

        print('The image has {} x {} voxels'.format(data.shape[0],data.shape[1]))
        data_downsampled = data[vals[2] - pad:vals[4] + pad, vals[1] - pad:vals[3] + pad]
        print('The downsampled image has {} x {} voxels'.format(
            data_downsampled.shape[0], data_downsampled.shape[1]))

        ds.PixelData = data_downsampled.tobytes()
        ds.Rows, ds.Columns = data_downsampled.shape

        if not os.path.exists(os.path.join(self.save,'T2', patient_dir)):
            os.mkdir(os.path.join(self.save,'T2', patient_dir))
        os.chdir(os.path.join(self.save,'T2',patient_dir))

        #save image
        if series == 't2':
            ds.save_as(patient_dir + '_' + str(index) + '_T2_' + vals[0] + '.dcm')

        return data_downsampled


    def order_dicom(self,dicom_file_list):
        '''
        As input, this method takes a list of paths to dicom directories (from find_dicom_paths), loads dicom, then orders them
        :param dicom_file_list
        :return list of files in correct order
        '''
        dicoms={}
        for path in dicom_file_list:
            file=path
            ds=pydicom.read_file(path)
            self.SHAPE=ds.pixel_array.shape
            dicoms[str(file)] = float(ds.SliceLocation)
        updated_imagelist=[key for (key, value) in sorted(dicoms.items(), key=lambda x: x[1])]
        return(updated_imagelist)


    def check_aligned_files(self,database,list_of_files):
        '''
        check for folder called 'align' in adc/highb
        param: list_of_files in mrn_scandate format, output of check_empty_files()
        :return: list of aligned files in mrn_scandate format
        '''
        #for development only
        aligned_files=[]
        for file in list_of_files:
            adc_output=os.listdir(os.path.join(self.basePATH,database,file,'dicoms','adc'))
            highb_output=os.listdir(os.path.join(self.basePATH,database,file,'dicoms','highb'))
            if 'aligned' in adc_output and 'aligned' in highb_output:
                aligned_files+=[file]
        print("there are a total of {} files aligned".format(len(aligned_files)))
        return(aligned_files)


    def check_empty_files(self):
        '''
        check for file completion in robtos_only and return only completed files with 'PIRADS' in voi file
        :return: list of patient names
        '''

        path = self.basePATH
        files = os.listdir(path)
        print("searching total of {} files in robots_only folder to see which ones are complete".format(len(files)))
        completed=[]
        for file in files:
            if len(file.split('_')) == 2:
                voi_dir = os.path.join(path, file, 'voi')
                adc_dir = os.path.join(path, file, r'dicoms\adc')
                highb_dir = os.path.join(path, file, r'dicoms\highb')
                t2_dir = os.path.join(path, file, r'dicoms\t2')

                voi_files = os.listdir(voi_dir)
                adc_files = os.listdir(adc_dir)
                highb_files = os.listdir(highb_dir)
                t2_files = os.listdir(t2_dir)

                #check for 'PIRADS' in voi files
                pirads_files=[]
                for file_voi in voi_files:
                    split_file = file_voi.replace(' ', '_').split('_')
                    if 'PIRADS' in split_file:
                        pirads_files+=[file_voi]

                if len(voi_files) > 0 and len(pirads_files)>0 and len(adc_files) > 1 and len(highb_files) > 1 and len(t2_files) > 1:
                    completed += [file]

        print('total of {} complete files with "PIRADS" in VOI file '.format(len(completed)))
        return(completed)


    def rescale_array(self,array):
        scaler=MinMaxScaler(feature_range=(0,255))
        scaler=scaler.fit(array)
        X_scaled=scaler.transform(array)
        return (X_scaled)


    def find_tumors_by_annotator(self,initials='bt'):
        '''parses through all databases and returns list of paths with tumors annotated by initials of annotator (saved
        after terminal underscore), then saves path to these .voi files
        note - if you want to update the databse, delete the file and re-run
        initials=initials of person doing annotation.  Should be terminal part
        '''

        if not os.path.exists(os.path.join(self.save_tumors,initials+'_annotated_tumor_paths.csv')):

            filepaths=[]
            counter=0
            for database in self.databases:
                patients_dirs=os.listdir(os.path.join(self.basePATH,database))
                for patient_dir in patients_dirs:
                    voi_files=os.listdir(os.path.join(self.basePATH,database,patient_dir,'voi'))
                    for file in voi_files:
                        split_file=file.split("_")
                        if len(split_file)>5 and split_file[-1]==initials+'.voi':
                            filepaths+=[os.path.join(self.basePATH,database,patient_dir,'voi',file)]
                            counter+=1
                            print(counter)

            filepaths_df=pd.DataFrame({'filepaths':filepaths})
            print('total of {} tumors for annotator {}'.format(len(filepaths),initials))
            filepaths_df.to_csv(os.path.join(self.save_tumors,initials+'_annotated_tumor_paths.csv'))


    def find_total_annotations(self,initials='bt'):
        '''parses through all databases and returns list of paths with tumors annotated by initials of annotator (saved
        after terminal underscore), then saves path to these .voi files
        note - if you want to update the databse, delete the file and re-run
        initials=initials of person doing annotation.  Should be terminal part
        '''

        filepaths=[]
        counter=0
        for database in self.databases:
            patients_dirs=os.listdir(os.path.join(self.basePATH,database))
            for patient_dir in patients_dirs:
                voi_files=os.listdir(os.path.join(self.basePATH,database,patient_dir,'voi'))
                for file in voi_files:
                    split_file=file.split("_")
                    if len(split_file) > 5 and split_file[-1] == initials + '.voi':
                        filepaths+=[os.path.join(self.basePATH,database,patient_dir,'voi',file)]
                        counter+=1
                        print(counter)

            filepaths_df=pd.DataFrame({'filepaths':filepaths})
            print('total of {} tumors for annotator {}'.format(len(filepaths),initials))
            filepaths_df.to_csv(os.path.join(self.save_tumors,initials+'_annotated_tumor_paths.csv'))




###############visualization tools##################

    def show_boxes(self):
        os.chdir(self.numpyPATH)
        files=os.listdir()
        for file in files:
            print(file)
            np_file=np.load(file)
            t2=np_file[0,:,:]; adc=np_file[1,:,:]; highb = np_file[2, :, :]
            #plt.imshow(t2, interpolation='nearest')
            #plt.imshow(adc, interpolation='nearest')
            #plt.imshow(highb, interpolation='nearest')
            plt.show()
            print(np_file)
            break


if __name__=='__main__':
    c=SegmentAligned()
    c.segment_aligned_by_annotator()
