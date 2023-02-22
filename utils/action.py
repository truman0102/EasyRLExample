import torch
import numpy as np


# 选择动作发生在与环境交互的过程中，按照贪婪策略选择动作，即选择Q值最大的动作，但是有一定的概率随机选择动作
def choose_action(model, state, epsilon, n_actions, device):
    if np.random.random() > epsilon:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        state = state.unsqueeze(0)
        actions = model.forward(state)
        action = torch.argmax(actions).item()
    else:
        action = np.random.choice(n_actions)
    return action


def choose_action_boltzmann(model, state, T, n_actions, device):
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float().to(device)
    state = state.unsqueeze(0)
    q_actions = model.forward(state) / T
    prob_actions = torch.softmax(q_actions, dim=1).detach().numpy()[0]
    action = np.random.choice(n_actions, p=prob_actions)
    return action
