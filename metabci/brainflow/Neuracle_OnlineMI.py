from neuracle_lib.dataServer import DataServerThread
import numpy as np
import mne  # 导入脑电处理mne包
import joblib
import threading    # 导入线程的包
import serial  # 导入串口的库
import time
import matplotlib.pyplot as plt
import serial.tools.list_ports
import pandas as pd
from metabci.brainda.algorithms.smooth_filter import smooth
'''
# =============================================================================
# 创建MNE.raw数据格式的info
# =============================================================================
'''
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg', 'eeg',  'eeg', 'eeg', 'eeg', 'eeg',
            'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg', 'eeg', 'eeg']
ch_names = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
            'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
            'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
            'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
            'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
            'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
            'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
            'Oz', 'O1', 'O2','ECG','HEOR','HEOL','VEOU','VEOL']

info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=ch_types)


# 数据读取类
# 包括 数据读取和分类的功能
class OnlineMI:
    def __init__(self,time_buffer, Stride, Name,CiShu,MoShi='ZhuaWo',
                 YuZhi=1, MIwindow=6, hostname='127.0.0.1', port=8712,srate=250,
                 drop_chan=['Fpz','Fp1','Fp2','AF7','AF8',
                                               'F7','F8',
                                               'FT7','FT8',
                                               'T7','T8',
                                               'TP7','TP8',
                                               'P7','P8',
                                               'PO7','PO8',
                                               'Oz', 'O1', 'O2','ECG','HEOR','HEOL','VEOU','VEOL']):

        self.hostname = hostname # 博睿康放大器接受数据IP地址，127.0.0.1代表本机地址
        self.port = port # 博睿康放大器接受在线数据的 port 号
        self.Name=Name # 被试名字，用于导入相应的分类器与生成保存数据的文件名字
        self.srate=250 # 采样率
        self.MoShi=MoShi # 在线控制模式，分为'ZhuaWo'和'duizhi'
        self.drop_chan=drop_chan # 删除的导联
        # 配置设备
        self.neuracle = dict(device_name='Neuracle', hostname=self.hostname, port=self.port,
                        srate=self.srate, chanlocs=['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
                                             'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                                             'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
                                             'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
                                             'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
                                             'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                                             'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
                                             'Oz', 'O1', 'O2' ,'ECG','HEOR','HEOL','VEOU','VEOL'] + ['TRG'], n_chan=65)  # 'TRG'为标签位,即使不打标签,也要加这个
       
        self.device = [self.neuracle] # 选着设备型号,默认Neuracle
        self.time_buffer = time_buffer  # 采集窗口大小为2s
        self.Stride=Stride # 设备采样率
        self.target_device = self.device[0]
        # 初始化 DataServerThread 线程
        self.thread_data_server = DataServerThread(device=self.target_device['device_name'], n_chan=self.target_device['n_chan'],
                                                   srate=self.target_device['srate'], t_buffer=self.time_buffer)
        self.s1 = threading.Semaphore(0)  # s1信号量的初始数量为0——开始采集信号量
        self.s2 = threading.Semaphore(0)  # s2信号量的初始数量为0——一次采集200ms数据，开始分类的信号量
        self.s3 = threading.Semaphore(1)  # s3信号量的初始数量为1——分类完成，准备采集数据的信号量
        self.s4 = threading.Semaphore(0)  # s4信号量的初始数量为0——串口输出指令控制外肢体的信号量
        self.s5 = threading.Semaphore(0)  # s5信号量的初始数量为0——一次分类完成，开始画图的信号量
        # 定义一个初始的变量，用于存放读取到的全部65导数据，包含标签
        self.data_transmit = np.zeros((self.neuracle['n_chan'], self.neuracle['srate']*self.time_buffer))
        # 定义一个初始的变量，用于存放读取到的全部64导数据，不包含标签
        self.Data_NoTRG = np.zeros((self.neuracle['n_chan']-1, self.neuracle['srate']*self.time_buffer))
        # 定义一个初始的变量，用于存放预处理后的，59导数据。
        # 因为在该类的多个函数中都运用到了上述三个数据变量，因此在这里定义self属性
        self.data_final = np.zeros((1, self.neuracle['n_chan']-1-len(self.drop_chan), self.neuracle['srate']*self.time_buffer))
        self.a = 0.05  # 分类结果的平滑因子
        self.pred_all=[] # 用于存放一个试次里所有的判别结果
        self.pred_all.append(np.array([1,])) # 因为控制策略的公式，这里需要初始化一个1值，代表“静息”
        self.YuZhi=YuZhi # 定义的触发阈值
        self.N=1 # 分类判别的次数
        self.flagstop =False # 采集停止的标志位
        self.ax=[] # 画图的横坐标
        self.ax.append(0)   # 初始化一个值，保证横坐标数的个数与pred_all里变量的个数一样。
        self.MIwindow=MIwindow # 运动想象窗口的大小
        self.CiShu=CiShu # 患者第几次来训练
        
        # 检验初始化是否成功
        if self.time_buffer< self.Stride:
            print(f'滑动步长大于窗口，请重新设置')
        else:
            print('OnlineMI类初始化成功')

   
    # 脑电信号预处理
    def preprocess(self,data):
        raw = mne.io.RawArray(data, info) # 将数据保存成mne格式
        raw.drop_channels(self.drop_chan)
        # raw.plot_psd()
        # raw = raw.notch_filter(np.arange(50, 125, 50)) # 陷波滤波
        raw = raw.filter(l_freq=8, h_freq=13) # 带通滤波
        raw = raw.set_eeg_reference(ref_channels='average') # 共参考
        data_process = raw.get_data() # 得到预处理后的数据
        return data_process


                
    def plot(self,x,y):
        plt.rcParams['figure.figsize'] = (20, 10)    # 图像显示大小
        plt.rcParams['font.sans-serif']=['SimHei']   # 防止中文标签乱码，还有通过导入字体文件的方法
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['lines.linewidth'] = 0.5        #设置曲线线条宽度   

        plt.clf()      
        plt.ion()# 清除之前的图
        #plt.subplot(2,1,1)
        plt.axhline(self.YuZhi,linestyle='dashed')
        plt.plot(x,y,'g-')     #等于agraghic.plot(ax,ay,'g-')
        plt.show()
        
                              

    # 建立脑电设备连接
    def device_connect(self):
        # 建立TCP/IP连接
        notconnect = self.thread_data_server.connect(hostname=self.target_device['hostname'], port=self.target_device['port'])
        if notconnect:
            raise TypeError("Can't connect recorder, Please open the hostport ")
        else:
            print('Data server connected')
            # 启动线程
            self.thread_data_server.Daemon = True
            self.thread_data_server.start()
            
            
    # 脑电数据读取
    def data_read(self):
        while True:
            self.s1.acquire()  # 向s1信号量请求一个信号
            self.flagstop = False # 采集标志位，可以开始采集信号
            # 这里resetData清零读取数据点数的标志位,是因为开启线程后一直读取脑电
            self.thread_data_server.ResetDataLenCount()  
            t_start = time.perf_counter() # 记录开始采集的时间点
            print('开始采集')
            
            while not self.flagstop: # 判断采集标志位，循环采集数据
                nUpdata = self.thread_data_server.GetDataLenCount()  # 得到读取到的点的个数
                
                # 如果读到200ms，即是250Hz采样率下的50个点，窗口是1s
                if nUpdata > (self.Stride * (self.target_device['srate'])-1):
                    self.thread_data_server.ResetDataLenCount()  # 这里resetData清零读取数据点数的标志位
                    self.N += 1
                    toc = time.perf_counter() # 记录当前采集的时间点，下面判断是否超过MIwindow
        
                    # 这里的if语句用来适应，外肢体的抓握模式和对指模式
                    if self.MoShi=='ZhuaWo':
                        # 如果到达time_buffer，则以后每200ms，存储一次数据，即窗口大小1s，步长200ms
                        if self.N >= (self.time_buffer/self.Stride) and toc-t_start < self.MIwindow:  
                            self.s3.acquire()  # 请求s3的一个信号
                            print(f'到达{self.time_buffer}s窗口')
                            self.data_transmit = self.thread_data_server.GetBufferData()  # 获取数据，数据长度=time_buffer*Fs
                            self.Data_NoTRG=self.data_transmit[0:self.neuracle['n_chan']-1,:] # 获取数据，去除标签TRG
                            self.s2.release()  # 释放s2一个信号，准备开始分类
                            print(f'释放s2信号，开始分类') 
                            
                        # 达到MIwindows的上限想象时间
                        if toc-t_start > self.MIwindow :   
                            print('任务超时，未满足阈值条件！')
                            # 定义一个df变量，将阈值，判别次数，真实判别值储存下来
                            dfData= {'YuZhi':self.YuZhi,'N':self.N,'ZhenShiZhi':self.pred_al,'MoShi':self.MoShi}
                            df_classfition = pd.DataFrame(data=dfData)
                            df_classfition.to_csv(self.Name+self.CiShu+'.csv',index=False,mode='a')
                            self.s4.release()  # 释放s4一个信号
                            self.pred_all=[] # 一个试次结束了，重置。清空上个试次的判别结果
                            self.pred_all.append(np.array([1,])) # 因为控制策略的公式，这里需要初始化一个1值，代表“静息”
                            self.ax=[] # 一个试次结束了，重置ax
                            self.ax.append(0)   #保存图1数据
                            self.flagstop = True  # 设置采集停止标志位
                            self.N=1 # 一个试次结束，重置N为1
                            
                    # 这里的if语句用来适应，外肢体的抓握模式和对指模式
                    # else后面执行对指模式
                    else:
                        # 如果到达time_buffer，则以后每200ms，存储一次数据，即窗口大小1s，步长200ms
                        if self.N >= (self.time_buffer/self.Stride):  
                            self.s3.acquire()  # 请求s3的一个信号
                            print(f'到达{self.time_buffer}s窗口')
                            self.data_transmit = self.thread_data_server.GetBufferData()  # 获取数据，数据长度=time_buffer*Fs
                            self.Data_NoTRG=self.data_transmit[0:self.neuracle['n_chan']-1,:]
                            # self.data_transmit = data[0:64, 0:200]
                            self.s2.release()  # 释放s2一个信号，准备开始分类
                            print(f'释放s2信号，开始分类')






    # 脑电数据进行分类
    def classify(self):
        ModelName=self.Name+'.pkl' # 模型的名字
        model = joblib.load(r"C:\Users\X1\Desktop\1/"+ModelName) # 导入模型
        # 设置循环，一次分类后接着下一次
        while True:
            self.s2.acquire()  # 请求s2的一个信号
            print('分类开始')
            data_preprocess = self.preprocess(self.Data_NoTRG)  # 对采集到的脑电信号进行预处理
            # 取预处理后的59导数据，并reshape成分类器的大小,这里-1是标签导，-5是去除的导联
            self.data_final= data_preprocess[0:self.neuracle['n_chan']-1-len(self.drop_chan),:].reshape(1, self.neuracle['n_chan']-1-len(self.drop_chan), self.neuracle['srate']*self.time_buffer)
            p_label = model.predict(self.data_final)  # 200ms预测一次结果
            
            self.pred_al = smooth(p_label[0])
            #self.pred_al = self.a * p_label + (1 - self.a) * self.pred_all[-1]  # 计算实时判别的结果，基于EEG的不稳定性，对输出的指令进行平滑建模处理
            self.pred_all.append(self.pred_al) # 存储当前分类结果
            self.ax.append(self.N)  # 存储当前分类的次数，作为画图的横坐标
            
            # self.s5.release()  # 释放s5的一个信号，开始画图
            
            self.plot(self.ax,self.pred_all) # 实时画图
            
            
            self.pred_al_send = round(self.pred_al[0]*100) # 这里是为了将判别处的标签结果转化为串口发送的字符信号，
                                                           # 使刺激界面可以按判别结果显示图片
                                                           # round()是为了取整数，[0]是因为pred_al为一个list数组，取第一个，不然数据结构不符
                                                           # *100 是为了等比放大，只有100-200的整数
            
            self.ser_read.write(str(self.pred_al_send).encode('utf-8')) # 串口发送指令
                                                                        # str()转换为串口可以发送的字符串，encode() 以十进制表示
            print(str(self.pred_al_send).encode('utf-8'))


            # 如果当前判别结果大于阈值，则控制外肢体运动
            if self.pred_al > self.YuZhi:
                print('满足阈值条件！')
                dfData= {'YuZhi':self.YuZhi,'N':self.N,'ZhenShiZhi':self.pred_al,'MoShi':self.MoShi}
                df_classfition = pd.DataFrame(data=dfData)
                df_classfition.to_csv(self.Name+self.CiShu+'.csv',index=False,mode='a')
                
                
                self.s4.release()  # 释放s4一个信号，串口控制外肢体
                self.pred_all=[] 
                self.pred_all.append(np.array([1,]))
                self.ax=[]
                self.ax.append(0)   
                self.flagstop = True  # 设置采集停止标志位
                self.N=1

            self.s3.release()  # 释放s3的一个信号，开始下一个200ms数据的获取
           
                 


          
                
               
 
    def serport(self,Ser_read,Ser_write,
              read_rcv=b'\x01\xE1\x01\x00\x05\r\n',
              write_rcv=b'\x41\x54\x2B\x44\x61\x74\x61\x31\xF1\xDD\x08\xAB\xC1\xAB\xFC\x20\x10\x11\x80\x0D\x0A'):
        
        read_rcv=read_rcv # 定义读取的指令
        write_rcv=write_rcv # 定义输出控制外肢体的指令
        port_list = list(serial.tools.list_ports.comports())
        if len(port_list) == 0:
            print('找不到串口')
        else:
            for i in range(0,len(port_list)):
                print(port_list[i])
                
        self.ser_read=serial.Serial(Ser_read, 9600) # COM口,波特率根据实际情况
        self.ser_write=serial.Serial(Ser_write, 9600) # COM口,波特率根据实际情况
        while True:
            rcv = self.ser_read.readline()  # 这个函数是运行后直接在此挂起，直到有串口数据进入,
                                        # 但是该函数需要，在发送端加上 r\n ，代表一行的结束   
            #rcv = ser_read.read_all() # 这个函数是读取目前缓存区的所有数据，但循环时出现问题   
            self.ser_read.flushInput()      # 清空缓存区
            # 通过if判断串口是否读到了指定数据,若读到则开始获取进来的脑电数据,进行分析
            if rcv==read_rcv:
                self.ser_read.write(str(round(self.YuZhi*100)).encode('utf-8')) # 串口发送指令，“YuZhi”的值，用来定义刺激界面图片的位置
                                                                                # 这里必须round，不然会有小数点，影响刺激界面后续的串口字符接收
                                                                                # 因为，刺激界面设置了每次仅读取3位字节，如果这里有小数就超过了三位字节，后面则从小数点后开始接收
                # print(str(self.YuZhi*100).encode('utf-8'))
                self.s1.release()  # 请求s1的一个信号 开始试次 准备采集数据
                print('MI试次开始')
            self.s4.acquire() # 请求s4分类完得信号，准备串口出指令

            # 抓握模式下: 输出指令弯曲，停止6s后，输出指令打开
            if  self.MoShi=='ZhuaWo':
                if self.pred_al > self.YuZhi:
                    i=222
                    self.ser_read.write(str(i).encode('utf-8')) # 这里串口发数据，是用来定义刺激界面放“真棒”还是“再接再厉”
                    print(str(i).encode('utf-8'))
                    self.ser_write.write(write_rcv) 
                    print('输出指令控制外肢体弯曲')
                    time.sleep(self.MIwindow)
                    self.ser_write.write(write_rcv) 
                    print('输出指令控制外肢体打开')
                else:
                    i=888
                    self.ser_read.write(str(i).encode('utf-8'))
                    print(str(i).encode('utf-8'))
                    
            # 抓握模式下: 输出指令对指，停止2s，等待完成对指任务
            else:
                if self.pred_al > self.YuZhi:
                    i=222
                    self.ser_read.write(str(i).encode('utf-8')) # 这里串口发数据，使刺激界面结束 实时外肢体反馈
                    print(str(i).encode('utf-8'))
                    self.ser_write.write(write_rcv) 
                    print('输出指令控制外肢体对指')
                    time.sleep(3)





        