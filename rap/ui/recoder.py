
import numpy as np

class Recoder:
    def __init__(self):
        self.data = []

    def step(self, data_t):
        self.data.append(data_t)

    def get_data(self):
        return self.data

    # def save_data(self):
    #     np.save
