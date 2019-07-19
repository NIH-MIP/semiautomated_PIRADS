import os
import pandas as pd

#author @t_sanf


class ImportVOI:

    def __init__(self):
        self.basePATH=r'T:\Automated_PIRADS_database\prostateX_vois'
        self.filename='ProstateX-2-Images-Train.csv'

    def prostateX_vois(self):
        '''
        reads in csv file 'ProstateX-Images-Train' and creates .voi files as points
        :return:
        '''

        file=pd.read_csv(os.path.join(self.basePATH,self.filename))

        df_lesion=pd.DataFrame()
        for i in range(file.shape[0]):
            if file.loc[i,'Name']=='t2_tse_tra0':
                df_line=file.iloc[i,:]
                df_lesion=df_lesion.append(df_line)
        print(df_lesion.loc[:,'ProxID'])
        print(df_lesion.index)


        for j in df_lesion.index:
            name=df_lesion.loc[j,'ProxID']
            list=df_lesion.loc[j,'ijk'].split(' ')
            x=list[0]; y=list[1]; z_slice=int(list[2])-1
            self.make_voi(pt=name,x=x,y=y,z_slice=z_slice,num=j)


    def make_voi(self,pt='mr_awesome',x=158,y=160,z_slice=9,num=1):

        file = open(os.path.join(self.basePATH,'vois',pt+'_biopsy_point'+'_'+str(num)+'.voi'), 'w')
        file.write('MIPAV VOI FILE \n')
        file.write('0		# curvelement_type of the VOI <Source-image></Source-image><ViewId>0</ViewId><z-flipped>0</z-flipped>\n')
        file.write('255		# color of VOI - red component\n')
        file.write('0		# color of VOI - green component\n')
        file.write('0		# color of VOI - blue component\n')
        file.write('255		# color of VOI - alpha component\n')
        file.write('1		# number of slices for the VOI\n')
        file.write(str(z_slice)+'		# slice number\n')
        file.write('1		# number of contours in slice\n')
        file.write('1  # number of pts in contour <Chain-element-type>1</Chain-element-type>\n')
        file.write(str(float(x))+' '+str(float(y))+'\n')
        file.write('1546112403  # unique ID of the VOI')
        file.close()

if __name__=='__main__':
    c=ImportVOI()
    c.prostateX_vois()


