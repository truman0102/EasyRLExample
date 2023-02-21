# EasyRLExample
basic RL examples in python

## agent
|| model-free      | model-based |
| ----------- | ----------- | ----------- |
|value|||
|policy|||

### Sarsa和Q学习的区别
Q学习并不需要知道下一时刻的动作，默认采取使Q取最大值的动作<br>
Sarsa下一时刻的动作是需要通过行为策略选取的，是下一个步骤一定会执行的动作

### 广义AC方法
价值函数的选取:
1. $\sum_{t=0}^{\infty}\Upsilon_t$,轨迹的总回报
2. $\sum_{t_`=t}^{\infty}\Upsilon_{t^`}$,动作后的回报
3. $\sum_{t_`=t}^{\infty}\Upsilon_{t^`}-b(s_t)$,baseline added
4. $Q^{\pi}(s_t,a_t)$,状态动作价值函数
5. $A^{\pi}(s_t,a_t)$,优势函数
6. $r_t+V_{\pi}(s_{t+1})-V_{\pi}(s_t)$,TD-error