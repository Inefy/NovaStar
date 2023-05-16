from collections import deque
import numpy as np
import random
from operator import itemgetter

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.epsilon = 1e-6  # small constant to avoid always excluding the minimum priority element

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        max_priority = max(self.priorities) if self.buffer else 1.0  # if the buffer is empty, the priority is 1
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        priorities = np.array(self.priorities)
        prob = priorities ** alpha
        prob /= prob.sum()  # Normalize to get the sampling probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)  # Sample indices with probability `prob`
        experiences = itemgetter(*indices)(self.buffer)  # Get corresponding experiences
        state, action, reward, next_state, done = zip(*experiences)

        # Compute importance-sampling weights
        weights = (len(self.buffer) * prob[indices]) ** -beta
        weights /= weights.max()  # Normalize weights

        return np.concatenate(state), action, reward, np.concatenate(next_state), done, indices, np.array(weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return len(self.buffer)
