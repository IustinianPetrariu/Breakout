from keras import models
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Flatten, InputLayer


class Model(tf.keras.Model):
    def __init__(self, num_states, num_actions, hidden_units=128):
        super().__init__(name = 'basic_ddqn')
        self.input_layer = InputLayer(input_shape=(num_states,))
        self.layer1 = Dense(hidden_units, activation = 'relu',kernel_initializer='RandomNormal')
        self.layer2 = Dense(hidden_units, activation = 'relu',kernel_initializer='RandomNormal')
        self.output_layer = Dense(num_actions, name='q_values')  #(None,num_actions)
        

       

    def call(self, inputs, training = None):
        x = self.input_layer(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


    def action_value(self, state):
        q_values = self.predict(state)
        best_action = np.argmax(q_values, axis = -1)
        return best_action[0], q_values[0]


