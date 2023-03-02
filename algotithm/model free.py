import numpy as np


action_dim = 4
state_num = 16
q_table = np.zeros((state_num, action_dim))  # 初始化Q表
p_table = np.random.dirichlet(np.ones(action_dim), size=state_num)  # 动作概率表格


# 蒙特卡洛方法
def mc_prediction(policy, env, num_episodes: int, discount_factor: float):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda time discount factor.
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple of np.arrays and the value is a float.
    """
    # The final value function
    V = {}
    # The number of returns for each state
    returns_count = {}
    # The sum of returns for each state over all sampled episodes
    returns_sum = {}
    # Implement this!
    for i in range(num_episodes):
        # 生成一个episode
        episode = []
        state = env.reset()
        while True:
            action = policy(state)  # 根据策略选择动作；也可以根据Q表选择动作
            next_state, reward, done, _ = env.step(action)  # 执行动作
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        # 计算Gt
        G = 0
        for state, action, reward in episode[::-1]:  # 轨迹从后往前
            G = reward + discount_factor * G  # Gt = Rt + λGt+1
            if state not in V:  # 如果状态不在V中，初始化
                V[state] = 0  # 初始化为0
                returns_count[state] = 0
                returns_sum[state] = 0
            returns_count[state] += 1  # 累计次数
            returns_sum[state] += G  # 累计Gt
            V[state] = returns_sum[state] / returns_count[state]  # V(s) = S(s)/N(s)
    return V

# 时序查分方法


def td_prediction(policy, env, num_episodes: int, discount_factor: float, alpha: float):
    """
    TD prediction algorithm. Calculates the value function
    for a given policy using sampling.
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple of np.arrays and the value is a float.
    """
    # The final value function
    V = {}
    # Implement this!
    for i in range(num_episodes):  # 生成一个episode
        state = env.reset()  # 初始化状态
        while True:
            action = policy(state)  # 根据策略选择动作；也可以根据Q表选择动作
            next_state, reward, done, _ = env.step(action)  # 执行动作
            if state not in V:  # 如果状态不在V中，初始化
                V[state] = 0  # 初始化为0
            # V(s) = V(s) + α[Rt+1 + λV(s+1) - V(s)]
            V[state] += alpha * (reward + discount_factor * V[next_state] - V[state])
            if done:
                break
            state = next_state  # 更新状态
    return V

# 动态规划方法


def dp_prediction(P: list, R: list, discount_factor: float, iteration_num: int):
    """
    DP prediction algorithm. Calculates the value function
    for a given policy using dynamic programming.
    Args:
        P: A list of transition probabilities for each state-action pair.
        R: A list of expected rewards for each state-action pair.
        discount_factor: Lambda time discount factor.
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple of np.arrays and the value is a float.
    """
    # The final value function
    V = {state: 0 for state in range(state_num)}
    for _ in range(iteration_num):
        for state in range(state_num):
            for action in range(action_dim):
                V[state] += p_table[state][action]*(R[state][action] + discount_factor*sum([P[state][action][next_state]*V[next_state] for next_state in range(state_num)]))


def policy(state,epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(np.arange(action_dim))
    else:
        return np.argmax(q_table[state])
    return np.random.choice(np.arange(action_dim), p=p_table[state])

def Sarsa(env, num_episodes: int, discount_factor: float, alpha: float, epsilon: float):
    """
    Sarsa algorithm. Calculates the optimal action-value function
    for a given epsilon-greedy policy.
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        episode_lengths is a list of integers, where each integer is the number of steps in each episode.
    """
    # The final action-value function.
    # Implement this!
    for i in range(num_episodes):
        # 同策略时序差分
        state = env.reset()
        action = policy(state, epsilon)
        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state, epsilon)
            q_table[state][action] += alpha * (reward + discount_factor * q_table[next_state][next_action] - q_table[state][action])
            if done:
                break
            state = next_state
            action = next_action
    return q_table

def Q_learning(env, num_episodes: int, discount_factor: float, alpha: float, epsilon: float):
    """
    Q-learning algorithm. Calculates the optimal action-value function
    for a given epsilon-greedy policy.
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        episode_lengths is a list of integers, where each integer is the number of steps in each episode.
    """
    # The final action-value function.
    # Implement this!
    for i in range(num_episodes):
        # 同策略时序差分
        state = env.reset()
        while True:
            action = policy(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            # Q-learning不需要next_action，直接选择最大的Q值
            q_table[state][action] += alpha * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action])
            if done:
                break
            state = next_state
    return q_table