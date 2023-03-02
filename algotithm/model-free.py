import numpy as np

# 蒙特卡洛方法
def mc_prediction(policy:function, env, num_episodes:int, discount_factor:float):
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
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        # 计算Gt
        G = 0
        for state, action, reward in episode[::-1]: # 轨迹从后往前
            G = reward + discount_factor * G # Gt = Rt + λGt+1
            if state not in V: # 如果状态不在V中，初始化
                V[state] = 0 # 初始化为0
                returns_count[state] = 0
                returns_sum[state] = 0
            returns_count[state] += 1 # 累计次数
            returns_sum[state] += G # 累计Gt
            V[state] = returns_sum[state] / returns_count[state] # V(s) = S(s)/N(s)
    return V

# 时序查分方法
# def td_prediction(policy:function, env, num_episodes:int, discount_factor:float, alpha:float):