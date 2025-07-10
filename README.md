# MetaBCI-SGXYLX

## 项目概述-“神工·灵犀指”脑控外肢体康复机器人
依托MetaBCI开源平台架构，“脑控外肢体康复机器人”项目集成了外肢体运动想象范式设计、模型校准、在线解码、视觉实时反馈、外肢体动触觉多模反馈等，实现了外肢体MI范式下实现超90%控制准确率及近乎0%的假阳性率。
- ① 在刺激呈现方面，调用brainstim子平台下的MI范式，并新增动图提示界面，实现更直观的MI刺激提示；
- ② 在实时控制方面，对brainflow子平台新增兼容博睿康NeuSen W脑电数据在线采集、处理分析、实时控制的功能函数Neuracle_OnlineMI；
- ③ 在信号处理方面，调用brainda子平台黎曼几何MDRM分类算法，进行离线校准及在线分析，新增brainda子平台解码结果平滑滤波Smooth函数，增加MI范式在线控制鲁棒性、可靠性；
- ④ 外设控制方面，实现了刺激界面实时视觉反馈以及外肢体机器人动、触觉反馈。

## 代码目录结构
总文件夹/ |-- demos/ | |-- demo_system/ | | |-- demo_TUNEXON_Calibration.py | | |-- demo_TUNEXON_Online.py | | |-- 脑控程序封装/ | | | |-- main.exe | | | |-- 相关程序文件 | | |-- 示例校准数据/ | | |-- 示例刺激界面/ | | |-- demo_stim/ | | | |-- demo_stim.py | | |-- demo_trans/ | | |-- demo_trans.py |-- metabci/ |-- LICENSE
