import os
import pandas as pd
from collections import Counter
from functools import reduce

#author @t_sanf


class ParseVOI(object):
    '''
    Convert VOI files to bounding boxes
    '''

    def __init__(self):
        '''
        :param filePATH (str) --> the path to the directory containing VOI files of interest
        '''
        self.PATH=''

    def list_of_dicts_all_images(self,filepaths):
        '''takes a

        :param filepaths --> list of filepaths
        :output
        '''

        for file in os.listdir(filepaths):
            one_dict=self.BBox_from_position(path=file)
            list_segment.append(one_dict)
        return(list_segment)


    def list_of_dicts(self,filepath):
        '''generates a list of dicts after iterating over stuff
        note - this function calls BBox_from_position function
        :return: list of dictionaries, one for each

        '''
        self.PATH=filepath
        file_path=filepath
        list_segment=[]
        for file in os.listdir(file_path):
            if file.endswith('.voi'):
                path_file=os.path.join(file_path, str(file))
                one_dict=self.BBox_from_position(path=path_file)
                list_segment.append(one_dict)
        #common = list(reduce(lambda x, y: x & y.keys(), list_segment))
        return(list_segment)


    def BBox_from_position(self,path):
        '''
        take output from get ROI_slice_loc and return category and bbox for each slice
        :return: dict {slice:(category,xmin,ymin,xmax,ymax)}
        '''

        pd_df=pd.read_fwf(path)

        # use get_ROI_slice_loc to find location of each segment
        dict=self.get_ROI_slice_loc(path=path)
        for slice in dict.keys():
            values=dict[slice]
            category=values[0]
            select_val=list(range(values[1],values[2]))
            specific_part=pd_df.iloc[select_val,:]
            split_df = specific_part.join(specific_part['MIPAV VOI FILE'].str.split(' ', 1, expand=True).rename(columns={0: "X", 1: "Y"})).drop(['MIPAV VOI FILE'], axis=1)

            # parse to find max/min X and Y values, save into dictionary
            xmin = int(float(split_df["X"].min()))
            ymin = int(float(split_df["Y"].min()))
            xmax = int(float(split_df["X"].max()))
            ymax = int(float(split_df["Y"].max()))
            tuple=(category,xmin,ymin,xmax,ymax)
            dict.update({slice:tuple})
        return(dict)


    def get_ROI_slice_loc(self,path=None):
        '''
        selects each slice number and the location of starting coord and end coord
        :return: dict of {slice number:(tuple of start location, end location)}

        '''

        pd_df=pd.read_fwf(path)

        #get the name of the file
        filename=path.split(os.sep)[-1].split('.')[0]

        #initialize empty list and empty dictionary
        slice_num_list=[]
        last_line=[]
        loc_dict={}

        #find the location of the last line -->

        for line in range(len(pd_df)):
            line_specific=pd_df.iloc[line,:]
            as_list=line_specific.str.split(r"\t")[0]
            if "# slice number" in as_list: #find location of all #slice numbers
                slice_num_list.append(line)
            if '# unique ID of the VOI' in as_list:
                last_line.append(line)

        for i in range(len(slice_num_list)):
            #for all values except the last value
            if i<(len(slice_num_list)-1):
                loc=slice_num_list[i]
                line_specific=pd_df.iloc[loc,:]
                slice_num=line_specific.str.split(r"\t")[0][0]
                start=slice_num_list[i]+3
                end=slice_num_list[i+1]-1
                loc_dict.update({slice_num:(filename,start,end)})

            #for the last value
            if i == (len(slice_num_list) - 1):
                loc = slice_num_list[i]
                line_specific=pd_df.iloc[loc,:]
                slice_num=line_specific.str.split(r"\t")[0][0]
                start=slice_num_list[i]+3
                end=(last_line[0]-1)
                loc_dict.update({slice_num: (filename, start, end)})

        return(loc_dict)


def intersection(list1, list2):
    # Use of hybrid method
    temp = set(list2)
    lst3 = [value for value in list1 if value in temp]
    return lst3


if __name__=='__main__':
    c=ParseVOI()
    c.list_of_dicts(r'S:\MIP\MRIClinical\anonymous_database\surgery_cases\5070947290\voi\wp_bt.voi')


