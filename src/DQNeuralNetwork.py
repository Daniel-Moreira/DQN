from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNeuralNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.model = Sequential([
            Conv2D(
                input_shape=self.state_size,
                filters=32, 
                kernel_size=[8, 8],
                strides=[4, 4],
                padding='VALID',
                activation='elu',
                kernel_initializer='GlorotUniform'
                ),
            Conv2D(
                filters=64, 
                kernel_size=[4, 4],
                strides=[2, 2],
                padding='VALID',
                activation='elu',
                kernel_initializer='GlorotUniform'
                ),
            Conv2D(
                filters=64, 
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='VALID',
                activation='elu',
                kernel_initializer='GlorotUniform'
                ),
            Flatten(),
            Dense(units=512,
                activation='elu',
                kernel_initializer='GlorotUniform'
                ),
            Dense(
                units=self.action_size,
                kernel_initializer='GlorotUniform',
                activation=None
                )
        ])
        
        self.optimizer = Adam(learning_rate=self.learning_rate)

        self.model.compile(optimizer=self.optimizer, loss='mse')

    def predict(self, state, batch_size=None):
        return self.model(state, training=False)

    def fit(self, states, targetQs, callbacks, batch_size, epochs, verbose):
        return self.model.fit(states, targetQs, callbacks=callbacks, batch_size=batch_size, verbose=verbose)

    def save_weights(self, path, name, index):
        file_path = f'{path}/{name}_{index}.ckpt'
        self.model.save_weights(file_path)

    def load_weights_from_file(self, path):
        self.model.load_weights(path)