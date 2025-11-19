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
        
    def activation_function(self, x):
        """
        Apply the selected activation function
        :param x: Input array
        :return: Activated output
        """
        if self.fact == 'sigmoid':
            return self.sigmoid(x)
        elif self.fact == 'relu':
            return self.relu(x)
        elif self.fact == 'linear':
            return self.linear(x)
        elif self.fact == 'tanh':
            return self.tanh(x)
        else:
            raise ValueError(f"Unknown activation function: {self.fact}")
    
    def activation_derivative(self, x):
        """
        Apply the derivative of the selected activation function
        :param x: Input array
        :return: Derivative of the activated output
        """
        if self.fact == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.fact == 'relu':
            return self.relu_derivative(x)
        elif self.fact == 'linear':
            return self.linear_derivative(x)
        elif self.fact == 'tanh':
            return self.tanh_derivative(x)
        else:
            raise ValueError(f"Unknown activation function: {self.fact}")
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # To prevent overflow, clip x to a reasonable range
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid activation function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function"""
        return (x > 0).astype(float)
    
    def linear(self, x):
        """Linear activation function"""
        return x
    
    def linear_derivative(self, x):
        """Derivative of linear activation function"""
        return np.ones_like(x)
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh activation function"""
        return 1 - np.tanh(x) ** 2
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network
        :param X: Input data of shape (n_samples, n_features)
        :return: Output of the network
        """
        # Initialize the first layer's activation with input data
        self.xi[0] = X.T if len(X.shape) > 1 else X.reshape(-1, 1)  # Transpose to match (n_features, n_samples)
        
        # Propagate through each layer from 1 to L-1 (excluding input layer)
        for l in range(1, self.L):
            # Calculate the weighted input (h) for layer l
            # h[l] = w[l] * xi[l-1] + theta[l] (broadcasting handles the bias addition)
            self.h[l] = np.dot(self.w[l], self.xi[l-1]) + self.theta[l].reshape(-1, 1)
            
            # Apply activation function to get activations for layer l
            # Using the activation function specified by self.fact
            self.xi[l] = self.activation_function(self.h[l])
        
        # Return the output of the last layer (transposed back to (n_samples, n_outputs))
        return self.xi[self.L-1].T