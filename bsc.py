import numpy as np

class BinarySymmetricChannel:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def receive(self, input):
        if np.random.random() < self.epsilon:
            return (input + 1) % 2
        
        return input

    def transitionProbability(self, input, output):
        return 1 - self.epsilon if output==input else self.epsilon