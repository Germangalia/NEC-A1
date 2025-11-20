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
    
    def backward_propagation(self, X, y):
        """
        Perform backward propagation algorithm
        :param X: Input data of shape (n_samples, n_features)
        :param y: Target values of shape (n_samples, n_outputs)
        """
        # Ensure y is in the right shape
        y = y.T if len(y.shape) > 1 else y.reshape(-1, 1)  # Transpose to match (n_outputs, n_samples)
        
        # Forward propagate to compute activations
        _ = self.forward_propagation(X)
        
        # Calculate output layer error (delta)
        # For the output layer (L-1), delta = (predicted - actual) * derivative of activation
        self.delta[self.L-1] = (self.xi[self.L-1] - y) * self.activation_derivative(self.h[self.L-1])
        
        # Backpropagate the error through the layers (from L-2 down to 1)
        for l in range(self.L-2, 0, -1):
            # Calculate error for layer l
            # delta[l] = (w[l+1]^T * delta[l+1]) * derivative of activation at layer l
            self.delta[l] = np.dot(self.w[l+1].T, self.delta[l+1]) * self.activation_derivative(self.h[l])
    
    def update_weights_and_thresholds(self):
        """
        Update weights and thresholds using calculated deltas and momentum
        """
        # Update weights and thresholds for each layer from 1 to L-1 (input layer doesn't have weights)
        for l in range(1, self.L):
            # Calculate weight changes: d_w[l] = -learning_rate * (delta[l] * xi[l-1]^T) 
            # This is the gradient descent update without momentum for now
            self.d_w[l] = -self.learning_rate * np.dot(self.delta[l], self.xi[l-1].T)
            
            # Add momentum term: d_w[l] = d_w[l] + momentum * d_w_prev[l]
            # Ensure the dimensions match before adding
            self.d_w[l] += self.momentum * self.d_w_prev[l]
            
            # Update weights: w[l] = w[l] + d_w[l]
            self.w[l] += self.d_w[l]
            
            # Calculate threshold (bias) changes: d_theta[l] = -learning_rate * delta[l]
            self.d_theta[l] = -self.learning_rate * self.delta[l]
            
            # Reshape d_theta to match the stored array dimensions if needed
            if self.d_theta[l].shape != self.d_theta_prev[l].shape:
                self.d_theta[l] = self.d_theta[l].reshape(self.d_theta_prev[l].shape)
            
            # Add momentum term: d_theta[l] = d_theta[l] + momentum * d_theta_prev[l]
            self.d_theta[l] += self.momentum * self.d_theta_prev[l]
            
            # Update thresholds: theta[l] = theta[l] + d_theta[l]
            self.theta[l] += self.d_theta[l]
            
            # Store current changes for next iteration's momentum
            self.d_w_prev[l] = self.d_w[l].copy()
            self.d_theta_prev[l] = self.d_theta[l].copy()
    
    def fit(self, X, y, epochs=100, validation_split=0.0):
        """
        Train the neural network with training and optional validation data
        :param X: Input data of shape (n_samples, n_features)
        :param y: Target values of shape (n_samples, n_outputs)
        :param epochs: Number of training epochs
        :param validation_split: Fraction of data to use for validation (between 0 and 1)
        :return: Training and validation loss history
        """
        import numpy as np
        
        # Convert inputs to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)
        
        # Determine the number of samples
        n_samples = X.shape[0]
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Split the data for training and validation if validation_split > 0
        if validation_split > 0.0 and validation_split < 1.0:
            val_size = int(n_samples * validation_split)
            X_val, y_val = X_shuffled[:val_size], y_shuffled[:val_size]
            X_train, y_train = X_shuffled[val_size:], y_shuffled[val_size:]
        else:
            # Use all data for training if no validation
            X_train, y_train = X_shuffled, y_shuffled
            X_val, y_val = None, None
        
        # Initialize loss history arrays
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle training data for each epoch
            train_indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[train_indices]
            y_train_shuffled = y_train[train_indices]
            
            # Calculate and store the training loss before this epoch's updates
            train_pred = self.predict(X_train)
            train_loss = self._mean_squared_error(y_train, train_pred)
            self.train_loss_history.append(train_loss)
            
            # Perform forward and backward propagation for each sample in the training batch
            for i in range(len(X_train)):
                # Forward propagation
                _ = self.forward_propagation(X_train_shuffled[i:i+1])
                
                # Backward propagation
                self.backward_propagation(X_train_shuffled[i:i+1], y_train_shuffled[i:i+1])
                
                # Update weights and thresholds
                self.update_weights_and_thresholds()
            
            # Calculate and store validation loss if validation data is available
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self._mean_squared_error(y_val, val_pred)
                self.val_loss_history.append(val_loss)
            else:
                # If no validation data, append NaN or the same as training loss
                self.val_loss_history.append(np.nan)
            
            # Print progress every 10% of epochs
            if (epoch + 1) % max(1, epochs // 10) == 0:
                val_loss_str = f"{self.val_loss_history[-1]:.6f}" if not np.isnan(self.val_loss_history[-1]) else 'N/A'
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.6f}, "
                      f"Validation Loss: {val_loss_str}")
        
        return self.train_loss_history, self.val_loss_history
    
    def _mean_squared_error(self, y_true, y_pred):
        """
        Calculate mean squared error
        :param y_true: True target values
        :param y_pred: Predicted values
        :return: Mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def predict(self, X):
        """
        Make predictions using the trained neural network
        :param X: Input data of shape (n_samples, n_features)
        :return: Predicted values of shape (n_samples, n_outputs)
        """
        # Convert to numpy array if not already
        X = np.array(X)
        
        # Perform forward propagation to get predictions
        predictions = self.forward_propagation(X)
        
        # Return the predictions, ensuring they're the right shape
        return predictions if predictions.shape[0] > 1 else predictions.ravel()
    
    def loss_epochs(self):
        """
        Return the evolution of training and validation errors for each epoch
        This method should be called after the fit method to get the stored loss history
        :return: 2 arrays of size (n_epochs, 2) containing training and validation error evolution
        """
        # The method returns 2 arrays of size (n_epochs, 2)
        # This likely means each array has shape (n_epochs, 2) where:
        # - Column 0 contains the error values
        # - Column 1 could contain additional metrics or be unused
        
        if hasattr(self, 'train_loss_history') and hasattr(self, 'val_loss_history'):
            n_epochs = len(self.train_loss_history)
            
            # Create arrays of shape (n_epochs, 2) as required
            # Column 0: error values, Column 1: can be used for additional metrics if needed
            train_errors = np.zeros((n_epochs, 2))
            val_errors = np.zeros((n_epochs, 2))
            
            # Fill the first column with the actual loss values
            train_errors[:, 0] = self.train_loss_history
            val_errors[:, 0] = self.val_loss_history
            
            return train_errors, val_errors
        else:
            # If no loss history exists, return empty arrays with the right shape
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)