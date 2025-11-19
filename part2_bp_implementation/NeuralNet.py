import numpy as np

class NeuralNet:
    def __init__(self, layers, learning_rate=0.01, momentum=0.0, fact='sigmoid'):
        """
        Initialize the neural network
        :param layers: Array with number of units in each layer, e.g. [4, 9, 5, 1] means 4 layers with 4 input units and 1 output unit
        :param learning_rate: Learning rate
        :param momentum: Momentum term
        :param fact: Activation function ('sigmoid', 'relu', 'linear', 'tanh')
        """
        # L: Number of layers
        self.L = len(layers)
        # n: Array with the number of units in each layer
        self.n = layers.copy()
        
        # h: Array of arrays for the fields (weighted inputs)
        self.h = []
        for lay in range(self.L):
            self.h.append(np.zeros(layers[lay]))
        
        # xi: Array of arrays for the activations
        self.xi = []
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))
            
        # w: Array of matrices for the weights
        self.w = []
        self.w.append(np.zeros((1, 1)))  # Placeholder, w[1] is the first actual weight matrix
        for lay in range(1, self.L):
            # w[lay] is the weight matrix from layer lay-1 to layer lay
            self.w.append(np.random.randn(layers[lay], layers[lay - 1]) * 0.5)
        
        # theta: Array of arrays for the thresholds (biases)
        self.theta = []
        for lay in range(self.L):
            self.theta.append(np.zeros(layers[lay]))
            
        # delta: Array of arrays for the propagation of errors
        self.delta = []
        for lay in range(self.L):
            self.delta.append(np.zeros(layers[lay]))
            
        # d_w: Array of matrices for the changes of the weights
        self.d_w = []
        self.d_w.append(np.zeros((1, 1)))  # Placeholder
        for lay in range(1, self.L):
            self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))
            
        # d_theta: Array of arrays for the changes of the thresholds
        self.d_theta = []
        for lay in range(self.L):
            self.d_theta.append(np.zeros(layers[lay]))
            
        # d_w_prev: Array of matrices for the previous changes of the weights (for momentum term)
        self.d_w_prev = []
        self.d_w_prev.append(np.zeros((1, 1)))  # Placeholder
        for lay in range(1, self.L):
            self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))
            
        # d_theta_prev: Array of arrays for the previous changes of the thresholds (for momentum term)
        self.d_theta_prev = []
        for lay in range(self.L):
            self.d_theta_prev.append(np.zeros(layers[lay]))
            
        # Activation function name
        self.fact = fact
        self.learning_rate = learning_rate
        self.momentum = momentum