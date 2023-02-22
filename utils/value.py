import torch
import numpy as np


def compute_target(gamma, v_final, r_lst, mask_lst):
    G = v_final
    td_target = []

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        # 从后往前计算
        G = r + gamma * G * mask
        td_target.append(G)
    return torch.from_numpy(np.array(td_target[::-1])).float()
    return torch.tensor(td_target[::-1]).float()  # 从前往后返回
