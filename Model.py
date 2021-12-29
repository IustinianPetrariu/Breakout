from keras import models
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Flatten, InputLayer,Dropout


class Model(tf.keras.Model):
    def __init__(self, num_states, num_actions, num_neurons=128):
        super().__init__(name = 'NN')
        self.input_layer  = InputLayer(input_shape=(num_states,))
        self.layer1       = Dense(num_neurons, activation = 'relu',kernel_initializer='glorot_uniform')
        self.layer2       = Dense(num_neurons, activation = 'relu',kernel_initializer='glorot_uniform')
        self.output_layer = Dense(num_actions,  activation='linear')  #(None,num_actions)
        
       
    def call(self, input):
        x = self.input_layer(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


    def take_action(self, state):
        q_values = self.predict(state)
        best_action = np.argmax(q_values, axis = -1)
        return best_action[0], q_values[0]


