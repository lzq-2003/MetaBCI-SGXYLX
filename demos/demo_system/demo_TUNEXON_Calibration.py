from neuracle_lib.readbdfdata import readbdfdata
from tkinter import *
import numpy as np
import os
import joblib

# mne imports
import mne
from metabci.brainda.datasets.Trans_data import transform_data
from metabci.brainda.algorithms.utils.model_selection import generate_kfold_indices, match_kfold_indices
from matplotlib import pyplot as plt
# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
from metabci.brainda.algorithms.manifold import MDRM



chan=39   # chan
OneClass = 1
TwoClass = 0
Rest = [0,0]
DATA_SPLIT = .9
labeled_data = []
fft_data = []
labels = []
SRF_Data=[]
RF_Data=[]
Rest_Data=[]
SRF_ddd=[]
Data=[]
arr4d = np.random.random( (1,chan,250) )
arr4y = np.random.random( (1) )
num_all=0


def check_files_format(path):
     filename = []
     pathname = []
     if len(path) == 0:
          raise TypeError('please select valid file')

     elif len(path) == 1:
          (temppathname, tempfilename) = os.path.split(path[0])
          if 'edf' in tempfilename:
               filename.append(tempfilename)
               pathname.append(temppathname)
               return filename, pathname
          elif 'bdf' in tempfilename:
               raise TypeError('unsupport only one neuracle-bdf file')
          else:
               raise TypeError('not support such file format')

     else:
          temp = []
          temppathname = r''
          evtfile = []
          idx = np.zeros((len(path) - 1,))
          for i, ele in enumerate(path):
               (temppathname, tempfilename) = os.path.split(ele)
               if 'data' in tempfilename:
                    temp.append(tempfilename)
                    if len(tempfilename.split('.')) > 2:
                         try:
                              idx[i] = (int(tempfilename.split('.')[1]))
                         except:
                              raise TypeError('no such kind file')
                    else:
                         idx[i] = 0
               elif 'evt' in tempfilename:
                    evtfile.append(tempfilename)

          pathname.append(temppathname)
          datafile = [temp[i] for i in np.argsort(idx)]

          if len(evtfile) == 0:
               raise TypeError('not found evt.bdf file')

          if len(datafile) == 0:
               raise TypeError('not found data.bdf file')
          elif len(datafile) > 1:
               print('current readbdfdata() only support continue one data.bdf ')
               return filename, pathname
          else:
               filename.append(datafile[0])
               filename.append(evtfile[0])
               return filename, pathname

if __name__ == '__main__':

    root = Tk()
    root.withdraw()
    ## select bdf or edf file
    Name='DHX'  #被试姓名
    
    for i in range(1,2):  #(5,6)代表第5次    训练次数
        
        XunLianCiShu='第'+str(i)+'次'
        filePath = r"C:\Users\lzqyzcjwds\Desktop\新建文件夹\\" + Name +'\\'+ XunLianCiShu + '\\校准'       #读取路径
        
        dir_name=os.listdir(filePath)
                
        
        
        for ele in dir_name:
            
            filename1 = filePath + '\\' + ele +  '\\data.bdf'
            filename2 = filePath  + '\\' + ele +  '\\evt.bdf'
            path = [filename1, filename2]
            filename, pathname = check_files_format(path)
            eeg = readbdfdata(filename, pathname)   # 读取文件中的数据
            raw = mne.io.read_raw_bdf(os.path.join(filename1), preload=True,exclude=(['Fpz','Fp1','Fp2','AF7','AF8',
                                  'F7','F8',
                                  'FT7','FT8',
                                  'T7','T8',
                                  'TP7','TP8',
                                  'P7','P8',
                                  'PO7','PO8',
                                  'Oz', 'O1', 'O2','ECG','HEOR','HEOL','VEOU','VEOL']))  # 读取文件中的脑电数据,,exclude=(['ECG','HEOR','HEOL','VEOU','VEOL'])

            # raw_notch=raw.notch_filter(np.arange(50, 125, 50) ,fir_design='firwin')
            raw_filter=raw.filter(l_freq=8, h_freq=13)
            raw_avg_ref =  raw_filter.set_eeg_reference(ref_channels='average')
            # raw_downsampled = raw_avg_ref.resample(sfreq=250)

            event_id = dict(Rest=1,SRF=2)   # 记录运动想象和静息开始状态的标签值
            tmin, tmax = 0, 6      # 设置提取试次脑电的时间     min and max
            events = eeg['events']
            picks = mne.pick_types(raw_avg_ref.info, meg=False, eeg=True, stim=False, eog=False
                                   ,exclude='bads')
            
            epochs = mne.Epochs(raw_avg_ref, events, event_id, tmin, tmax, proj=False,
                        picks=picks, baseline=None, preload=True, verbose=False)
            labels = epochs.events[:, -1]
            Data = epochs.get_data()
            
            
            for n in range(tmin, tmax-1):
                X = Data[:, 0:chan, int(n*250):int((n+1)*250)]  # 滑动窗口，窗口为1s
                y = labels  # 标签
                np.transpose(y)
                arr4d = np.vstack((arr4d, X))  # 拼接脑电数据
                arr4y = np.hstack((arr4y, y))   # 拼接标签信息


    X, y, df = transform_data(arr4d, arr4y)
    
    # set_random_seeds(38)
    kfold = 5
    indices = generate_kfold_indices(df, kfold=kfold)

    estimator=MDRM()


    accs = []
    for k in range(kfold):
        train_ind, validate_ind, test_ind = match_kfold_indices(k, df, indices)
        # merge train and validate set
        train_ind = np.concatenate((train_ind, validate_ind))
        p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
        accs.append(np.mean(p_labels==y[test_ind]))
    print(np.mean(accs))
    # If everything is fine, you will get the accuracy about 0.875.
    
    
    
    names        = ['Rest','SRF']
    plt.figure(0)

    joblib.dump(estimator,  r'C:\Users\lzqyzcjwds\Desktop\新建文件夹\\'+ Name +'.pkl')     # 保存路径

    lr = joblib.load( r'C:\Users\lzqyzcjwds\Desktop\新建文件夹\\'+ Name +'.pkl')       # 保存路径
    ac = []
    y_pred = lr.predict(X[test_ind])
    ac.append(np.mean(y_pred==y[test_ind]))
    print(ac)