import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable


def smoother(alpha: float,
            initial_value: float,
            current_value: float,
            mode: str = 'standard',
            kalman_params: Optional[Dict[str, Any]] = None) -> Callable[[float], float]:
    """
    参数:
    alpha: 平滑因子 (0 < alpha < 1)
    initial_value: 初始预测值
    mode: 平滑算法选择 ('standard', 'differential', 'kalman')
    kalman_params: 卡尔曼滤波器参数 (仅当mode='kalman'时使用)

    返回:
    接受测量值，返回平滑后的预测值
    """
    # 初始化状态

    state = {
        'x': np.array([initial_value]),
        'F': np.eye(1),
        'Q': np.eye(1) * 1e-6,
        'H': np.eye(1),
        'R': np.eye(1) * (1 - alpha) / max(alpha, 1e-8),
        'P': np.eye(1) * 0.1
    }

    # 处理卡尔曼参数
    if mode == 'kalman' and kalman_params:
        for key in ['F', 'Q', 'H', 'R', 'P']:
            if key in kalman_params:
                state[key] = kalman_params[key]

    if mode == 'standard':
        # 标准指数平滑
        prediction = alpha * initial_value + (1 - alpha) * current_value

    elif mode == 'differential':
        # 差分形式
        prediction = current_value + alpha * (initial_value - current_value)

    elif mode == 'kalman':
        # 卡尔曼滤波形式
        # 预测步骤
        x_pred = state['F'] @ state['x']
        P_pred = state['F'] @ state['P'] @ state['F'].T + state['Q']

        # 更新步骤
        y = initial_value - state['H'] @ x_pred
        S = state['H'] @ P_pred @ state['H'].T + state['R']
        K = P_pred @ state['H'].T @ np.linalg.inv(S)

        # 状态更新
        state['x'] = x_pred + K @ y
        state['P'] = (np.eye(1) - K @ state['H']) @ P_pred
        prediction = np.array(state['x'][0])

    else:
        raise ValueError(f"未知的平滑模式: {mode}")


    return prediction

