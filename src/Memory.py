from collections import deque
from random import sample

class Memory:
    def __init__(self, size):
        self.memory = deque(maxlen = size)
    
    def sample(self, size):
        if size > len(self.memory):
            size = len(self.memory)

        return sample(self.memory, size)
        
    def store(self, data):
        self.memory.append(data)