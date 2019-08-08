import The_System as tsys
import pandas as pd
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam

class ReplayBuffer(object):
    def __init__(self max_size, input_shape, n_action, discrete=False):
        self.mem_size = max_size
        self

