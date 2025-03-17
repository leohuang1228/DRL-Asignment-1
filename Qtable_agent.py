import numpy as np
import pickle


class QLearningAgent:
    def __init__(self, action_size, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize Q-table with zeros
        self.q_table = {}

    def choose_action(self, state):
        self.add_new_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def choose_best_action(self, state):
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        self.add_new_state(next_state)
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + (0 if done else self.discount_factor * self.q_table[next_state][best_next_action])
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def add_new_state(self, state):
        if state in self.q_table.keys():
            return
        self.q_table[state] = np.zeros(self.action_size)

    def save_q_table(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_q_table(self, file_path):
        with open(file_path, 'rb') as file:
            self.q_table = pickle.load(file)
