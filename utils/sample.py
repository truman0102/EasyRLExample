import random
import numpy as np
from collections import deque, namedtuple

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done')
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity  # 缓冲区大小
        self.buffer = deque(maxlen=capacity)  # 缓冲区

    def push(self, transition):
        # transition = (state, action, next_state, reward, done)
        self.buffer.append(Transition(*transition))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)  # 返回5个元组，分别是state, action, reward, next_state, done
    
    @property
    def size(self):
        return len(self.buffer)