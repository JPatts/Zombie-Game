import numpy as np
from collections import deque
import pickle

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.alpha = 0.1
        self.alpha_decay = 0.995
        self.alpha_min = 0.01
        self.gamma = 0.95
        
        # Experience replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Metrics
        self.training_steps = 0
        self.episode_rewards = []

    def preprocess_state(self, state):
        """Normalize state values"""
        return tuple(np.round(state, 3))  # Reduce state space

    def get_action(self, state):
        state = self.preprocess_state(state)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Get Q-values for all actions
        q_values = [self.q_table.get((state, a), 0.0) for a in range(self.action_size)]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def update(self, state, action, reward, next_state, done):
        """Update Q-values and learn from experience"""
        # Store experience
        self.remember(state, action, reward, next_state, done)
        
        # Update current state-action pair
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        
        current_q = self.q_table.get((state, action), 0.0)
        next_max_q = max([self.q_table.get((next_state, a), 0.0) 
                         for a in range(self.action_size)])
        
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q
        
        # Batch learning from replay buffer
        self._replay()
        
        # Update parameters
        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if self.alpha > self.alpha_min:
                self.alpha *= self.alpha_decay
        
        self.training_steps += 1

    def _replay(self):
        """Learn from past experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            current_q = self.q_table.get((state, action), 0.0)
            if done:
                target_q = reward
            else:
                next_max_q = max([self.q_table.get((next_state, a), 0.0) 
                                for a in range(self.action_size)])
                target_q = reward + self.gamma * next_max_q
            
            new_q = current_q + self.alpha * (target_q - current_q)
            self.q_table[(state, action)] = new_q

    # This function will save the agent state and is very useful
    def save(self, filename):
        """Save agent state"""
        state = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'training_steps': self.training_steps
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    # Load can then be used to test a learned agent in a different state space
    def load(self, filename):
        """Load agent state"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        self.q_table = state['q_table']
        self.epsilon = state['epsilon']
        self.alpha = state['alpha']
        self.training_steps = state['training_steps']

    # Used for data aggregation within main 
    def get_stats(self):
        """Return current learning statistics"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'training_steps': self.training_steps
        }