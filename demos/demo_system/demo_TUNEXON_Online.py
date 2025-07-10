from metabci.brainflow.TUNEXON_OnlineMI import OnlineMI

if __name__ == '__main__':

    M=OnlineMI(time_buffer=1, Stride=0.2, Name='ZSH',YuZhi=1.5,
               MIwindow=6,MoShi='ZhuaWo',CiShu='Di 2 Ci')
    M.device_connect()  # 创建设备连接

    task1 = threading.Thread(target=M.data_read, name="data_read", daemon=True)  # 创建范式线程
    task2 = threading.Thread(target=M.classify, name='classify', daemon=True)   # 创建数据采集线程
    task3 = threading.Thread(target=M.serport, name='serport', daemon=True, args=('COM4','COM3'))  # 创建分类线程

    task1.start()   # 启动范式线程
    task2.start()   # 启动数据采集线程
    task3.start()   # 启动分类线程

    task1.join()
    task2.join()
    task3.join()
