import random
import numpy as np
from collections import deque

class ExperienceReplay:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque(maxlen = capacity)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        return random.sample(self.memory, self.batch_size)
    
    def size(self):
        return len(self.memory)

