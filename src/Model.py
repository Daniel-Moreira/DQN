from DQNeuralNetwork import DQNetwork
from Memory import Memory
from itertools import compress
import numpy as np
import tensorflow as tf
import os
import datetime
import pickle

MEMORY_SIZE = 160000
# MEMORY_SIZE = 4000
MODEL_SAVE_PATH = 'space_invaders/model/cp-{epoch:d}.cpkt'
MODEL_DIR = os.path.dirname(MODEL_SAVE_PATH)
TARGET_NETWORK_UPDATE = 2000

class Model:
    def __init__(self, state_size, action_size, learning_rate, discount_rate, batch_size):
        self.DQNetwork = DQNetwork(state_size, action_size, learning_rate)
        self.DQNetworkTarget = DQNetwork(state_size, action_size, learning_rate)

        self.episodes_memory = Memory(MEMORY_SIZE)

        self.action_size = action_size
        self.batch_size = batch_size
        self.discount_rate = discount_rate

        self.update_step = 0
        self.start_time = datetime.datetime.now()

        self.DQNetwork.load_weights_from_file(self.load_model())
        # self.DQNetworkTarget.load_weights_from_file(self.load_model())

    # Choose an action to apply in the current state. 
    def choose_action(self, state, epsilon, print_v=False):

        # Explore at random
        if np.random.rand() <= epsilon:
            return np.random.randint(0, self.action_size)

        # Choose best action
        q_values = self.DQNetwork.predict(np.array([state]))

        if print_v:
            print('Predicted q value: ', q_values)
        return np.argmax(q_values[0])

    def store_memory(self, state, action, reward, next_state, terminated):
        # Store the episode in memory
        self.episodes_memory.store((state, action, reward, next_state, terminated))

    # Evaluate result of taking an action
    def evaluate_action(self):        
        # Apply gradient descent
        self.NNtraining(self.batch_size)

    # Copy weights from target DQ to DQ
    def alighn_target_model(self):
        print('SYNCING NETWORKS')
        self.DQNetworkTarget.model.set_weights(self.DQNetwork.model.get_weights())

    # Neural Network training model with replay
    def NNtraining(self, batch_size):
        # Sample episodes from memory
        batch = self.episodes_memory.sample(batch_size)

        states = np.array([each[0] for each in batch], ndmin=3)
        next_states = np.array([each[3] for each in batch], ndmin=3)
        targetQs = self.DQNetwork.predict(states)
        nextQs = self.DQNetworkTarget.predict(next_states)
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch]).reshape(batch_size, 1)
        terminated = np.array([each[4] for each in batch])

        # Sum the rewards
        # 1. set t[:][actions] to zero = (targetQs - targetQs * actions)
        # 2. sum the rewards = (rewards * actions)
        targetQs = (targetQs - targetQs * actions) + (rewards * actions)

        # 1. Get max_a' from nextQ = np.max(nextQs, axis=1)
        # 2. Multiply by the states that aren't fineshed = np.max(nextQs, axis=1) * terminated 
        targetQs += ((np.max(nextQs, axis=1) * terminated * self.discount_rate).reshape(batch_size, 1)) * actions
        # print(f'Actions: {actions}, Reward: {rewards}, Target: {targetQs}')

        self.DQNetwork.fit(states, targetQs, callbacks=[], batch_size=len(batch), epochs=1, verbose=1)

        if self.update_step == TARGET_NETWORK_UPDATE:
            self.alighn_target_model()
            self.update_step = 0

        self.update_step+=1

        delta_time = round((datetime.datetime.now() - self.start_time).seconds/60)
        
        if (delta_time % 20 == 0):
            self.save_model(round(delta_time/20))

    # Save the model
    def save_model(self, time):
        self.DQNetwork.save_weights(MODEL_DIR, 'model', time)
        
    # Load a saved model
    def load_model(self):
        return tf.train.latest_checkpoint(MODEL_DIR)

    # returns the memory object
    def get_memory(self):
        return self.episodes_memory