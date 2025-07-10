from psychopy import monitors
import numpy as np
from metabci.brainstim.framework import Experiment
from metabci.brainstim.paradigm import MI,paradigm

if __name__ == "__main__":
    # 设置显示器参数
    mon = monitors.Monitor(name='PHL 223i5',        # 显示器名称
                           width=50,                # 显示器物理宽度(cm)
                           distance=60,             # 显示器距人眼距离(cm)
                           verbose=False)           #关闭冗余输出
    mon.setSizePix([1920, 1080])    # 设置显示器的分辨率
    mon.save()    # 保存显示器配置
    # 初始化实验对象
    bg_color_warm = np.array([0, 0, 0]) #背景颜色RGB值（范围[-1,1])
    win_size=np.array([1920, 1080])     #背景大小（像素）
    ex = Experiment(monitor=mon,
                    bg_color_warm=bg_color_warm,    # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
                    screen_id=0,
                    win_size=win_size,          # 范式边框大小(像素表示)， 默认[1920,1080]
                    is_fullscr=True,            # 全屏模式,此时 win_size 参数默认屏幕分辨率
                    record_frames=False,        # 不记录帧率（调试时可开启）
                    disable_gc=False,           # 不禁用垃圾回收（保持性能）
                    process_priority='normal',  # 进程优先级设置为normal（避免系统卡顿）
                    use_fbo=False)              # 不使用离屏渲染（普通刺激无需开启）

    # 获取实验窗口和基本参数设置
    win = ex.get_window()   # 获取之前创建的实验窗口对象
    fps = 60   # 设置屏幕刷新率为60Hz
    text_pos = (0.0, 0.0)   # 设置提示文本的坐标（屏幕中心）
    left_pos = [[-480, 0.0]]    # 左手刺激位置（水平向左480像素）
    right_pos = [[480, 0.0]]    # 右手刺激位置（水平向右480像素）
    tex_color = 2*np.array([179, 45, 0])/255-1  # 设置提示文本颜色（将RGB颜色转换为PsychoPy使用的[-1,1]范围）
    normal_color = [[-0.8,-0.8,-0.8]]   # 设置未激活状态默认颜色为灰色
    image_color = [[-1,-1,-1]]          # 设置刺激激活状态背景为原色
    symbol_height = 100                 # 设置提示符号的高度（像素）
    n_Elements = 1                      # 左右手各显示1个刺激元素
    stim_length = 256    # 设置刺激图像长度（像素）
    stim_width = 256     # 设置刺激图像宽度（像素）
    anim_speed = 0.5
    basic_MI = MI(win=win)
    basic_MI.config_color(refresh_rate=fps, text_pos=text_pos, left_pos=left_pos,
                          right_pos=right_pos, tex_color=tex_color,
                          normal_color=normal_color, image_color= image_color,
                          symbol_height=symbol_height, n_Elements=n_Elements,
                          stim_length=stim_length, stim_width=stim_width, anim_speed = anim_speed,)
    basic_MI.config_response()  # 配置响应机制

    # 注册运动想象范式
    bg_color = np.array([-1, -1, -1])   # 背景颜色（纯黑色）
    hand_mode = "left"  # 选择实验模式
    display_time = 1     # 初始显示时间
    rest_time = 7        # 休息间隔时间(秒)
    index_time = 1       # 指示时间(秒)
    image_time = 6       # 运动想象任务时间(秒)
    response_time = 2    # 响应等待时间(秒)
    port_addr = None     # 并口地址（用于发送触发标记）
    nrep = 10            # 每组实验的重复次数
    lsl_source_id = None    # LSL数据流ID（用于脑电同步）
    online = True      # 是否在线模式（False表示离线实验）
    device_type = "Neuracle"
    ex.register_paradigm('SFMI', paradigm,
                     VSObject=basic_MI, bg_color=bg_color, hand_mode=hand_mode,
                     display_time=display_time, index_time=index_time,
                     rest_time=rest_time, response_time=response_time,
                     port_addr=port_addr, nrep=nrep, image_time=image_time,
                     pdim='mi',lsl_source_id=lsl_source_id, online=online, device_type=device_type)

    ##启动实验
    ex.run()