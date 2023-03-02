import torch
import numpy as np


def compute_target(gamma, r_lst, done_lst, v_final=0, ) -> torch.tensor():
    G = v_final  # G是最后一个状态的V值
    td_target = []
    for r, done in zip(r_lst[::-1], done_lst[::-1]):
        # 从后往前计算
        G = r + gamma * G * (1 - done)
        td_target.append(G)
    return torch.from_numpy(np.array(td_target[::-1])).float()  # 从前往后返回
