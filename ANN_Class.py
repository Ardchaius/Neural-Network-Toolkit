# Neural network implementation consisting of multiple subclasses

import numpy as np
import ANN_Learning_Rate_Decays as lrd


# Basic Neural Network, with some regularization techniques available
class ANN:
    weights = []  # List of weight matrix arrays
    biases = []  # List of bias matrix arrays
    layer_dims = []  # List of number of nodes in each layer
    num_layers = 0  # Number of layers
    learning_rate = 0  # Learning rate
    act_hid = 0  # Activation function for hidden layers
    act_out = 0  # Activation function for output layer
    cost = 0  # Cost function
    show_error = 0  # Constant for how often to display accuracy and cost
    train_x = []  # Training input values
    train_y = []  # Training actual values, goal values
    test_x = []  # Test input values
    test_y = []  # Test actual values
    keep_chance = 1  # Dropout keep percentage
    dp_vectors = []  # list of dropout vectors
    weight_decay = 0  # Weight decay constant
    lr_decay_method = 0  # Learning rate decay method to be use
    lr_initial = 0  # Initial learning rate
    batch_size = 0  # Mini-batch size
    beta1, beta2 = 0, 0  # Constants for Adam optimization
    epsilon = 0  # Constant for Adam optimization
    vdw = []  # List of matrices for Adam values
    sdw = []
    vdb = []
    sdb = []
    adam_count = 0  # Count of how many Adam steps have been performed

    # Initialization function
    def __init__(self, layer_dims, learning_rate, train_x, train_y, act_hid, act_out, cost, batch_size=1,
                 show_error=0, test_x=[], test_y=[], keep_chance=1, weight_decay=0, beta1=0, beta2=0, epsilon=0,
                 lr_decay_method=lrd.Constant(0)):
        self.num_layers = len(layer_dims)
        self.layer_dims = layer_dims
        self.lr_initial = learning_rate
        self.show_error = show_error
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.act_hid = act_hid
        self.act_out = act_out
        self.cost = cost
        self.keep_chance = keep_chance
        self.weight_decay = weight_decay
        self.lr_decay_method = lr_decay_method
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.reset()

    # Function for resetting weights and biases
    def reset(self):
        self.weights = [None] + [np.random.randn(self.layer_dims[layer], self.layer_dims[layer - 1]) *
                                 np.sqrt(2 / self.layer_dims[layer - 1]) for layer in range(1, self.num_layers)]
        self.biases = [None] + [np.zeros((self.layer_dims[layer], 1)) for layer in range(1, self.num_layers)]
        self.dp_vectors = [None] + [np.zeros((self.layer_dims[layer], 1)) for layer in
                                    range(1, self.num_layers - 1)] + [None]
        self.adam_count = 0
        self.vdw = [None] + [np.zeros((self.layer_dims[layer], self.layer_dims[layer - 1])) for layer in
                             range(1, self.num_layers)]
        self.sdw = [None] + [np.zeros((self.layer_dims[layer], self.layer_dims[layer - 1])) for layer in
                             range(1, self.num_layers)]
        self.vdb = [None] + [np.zeros((self.layer_dims[layer], 1)) for layer in range(1, self.num_layers)]
        self.sdb = [None] + [np.zeros((self.layer_dims[layer], 1)) for layer in range(1, self.num_layers)]

    # Forward propagation function
    def forward_propagation(self, input_data, keep_chance=1):
        acts = [[None] for i in range(self.num_layers)]
        z_values = [[None] for i in range(self.num_layers)]
        acts[0] = input_data

        for layer in range(1, self.num_layers):
            z_values[layer] = np.dot(self.weights[layer], acts[layer - 1]) + self.biases[layer]
            if layer == (self.num_layers - 1):
                acts[layer] = self.act_out.func(z_values[layer])
            else:
                self.dp_vectors[layer] = np.random.rand(self.dp_vectors[layer].shape[0], 1) < keep_chance
                acts[layer] = (self.dp_vectors[layer] * self.act_hid.func(z_values[layer])) / keep_chance
        return z_values, acts

    # Backward propagation function
    def backward_propagation(self, z_values, acts, actual):
        d_weights = [[None] for i in range(self.num_layers)]
        d_biases = [[None] for i in range(self.num_layers)]
        z_grad = self.cost.grad(acts[-1], actual) * self.act_out.grad(z_values[-1])
        d_weights[-1] = np.dot(z_grad, acts[-2].T) + (self.weight_decay / actual.shape[1]) * self.weights[-1]
        d_biases[-1] = np.sum(z_grad, axis=1, keepdims=True)

        for layer in range(self.num_layers - 2, 0, -1):
            z_grad = (np.dot(self.weights[layer + 1].T, z_grad) * self.act_hid.grad(z_values[layer])) * self.dp_vectors[
                layer] / self.keep_chance
            d_weights[layer] = np.dot(z_grad, acts[layer - 1].T) + (self.weight_decay / actual.shape[1]) * self.weights[
                layer]
            d_biases[layer] = np.sum(z_grad, axis=1, keepdims=True)

        return d_weights, d_biases

    # Update parameters
    def update_parameters(self, d_weights, d_biases):
        if self.beta1 == 0:
            self.weights = [None] + [w - self.learning_rate * dw for w, dw in zip(self.weights[1:], d_weights[1:])]
            self.biases = [None] + [b - self.learning_rate * db for b, db in zip(self.biases[1:], d_biases[1:])]
        else:
            self.adam_count += 1
            self.vdw = [None] + [self.beta1 * vdwl + (1 - self.beta1) * dw for
                                 vdwl, dw in zip(self.vdw[1:], d_weights[1:])]
            self.sdw = [None] + [self.beta2 * sdwl + (1 - self.beta2) * (dw ** 2) for
                                 sdwl, dw in zip(self.sdw[1:], d_weights[1:])]
            self.vdb = [None] + [self.beta1 * vdbl + (1 - self.beta1) * db for
                                 vdbl, db in zip(self.vdb[1:], d_biases[1:])]
            self.sdb = [None] + [self.beta2 * sdbl + (1 - self.beta2) * (db ** 2) for
                                 sdbl, db in zip(self.sdb[1:], d_biases[1:])]
            vdw_corr = [None] + [vdwl / (1 - np.power(self.beta1, self.adam_count)) for vdwl in self.vdw[1:]]
            sdw_corr = [None] + [sdwl / (1 - np.power(self.beta2, self.adam_count)) for sdwl in self.sdw[1:]]
            vdb_corr = [None] + [vdbl / (1 - np.power(self.beta1, self.adam_count)) for vdbl in self.vdb[1:]]
            sdb_corr = [None] + [sdbl / (1 - np.power(self.beta2, self.adam_count)) for sdbl in self.sdb[1:]]
            self.weights = [None] + [w - self.learning_rate * (vdwl_corr / (np.sqrt(sdwl_corr) + self.epsilon)) for
                                     w, vdwl_corr, sdwl_corr in zip(self.weights[1:], vdw_corr[1:], sdw_corr[1:])]
            self.biases = [None] + [b - self.learning_rate * (vdbl_corr / (np.sqrt(sdbl_corr) + self.epsilon)) for
                                    b, vdbl_corr, sdbl_corr in zip(self.biases[1:], vdb_corr[1:], sdb_corr[1:])]

    # Learn from training data
    def learn(self, iterations):
        for iteration in range(iterations):
            self.learning_rate = self.lr_decay_method.decay(self.lr_initial, iteration)
            permutation = np.random.permutation(self.train_y.shape[1])
            x_permutated = self.train_x[:, permutation]
            y_permutated = self.train_y[:, permutation]
            for batch_start in range(0, self.train_y.shape[1], self.batch_size):
                x = x_permutated[:, batch_start:batch_start + self.batch_size]
                y = y_permutated[:, batch_start:batch_start + self.batch_size]
                z_values, acts = self.forward_propagation(x, keep_chance=self.keep_chance)
                d_weights, d_biases = self.backward_propagation(z_values, acts, y)
                self.update_parameters(d_weights, d_biases)

            if iteration % (self.show_error + 1) == 0:
                _, train_acts = self.forward_propagation(self.train_x)
                print('Cost: ',
                      self.cost.func(train_acts[-1], self.train_y) + np.sum([np.sum(w) for w in self.weights[1:]]) * (
                              self.weight_decay / (2 * self.train_y.shape[1])))
                print('Training Accuracy: ',
                      np.sum(np.round(train_acts[-1]) == self.train_y) * 100 / self.train_y.shape[1])
                if self.test_x != []:
                    _, test_acts = self.forward_propagation(self.test_x)
                    print('Test Accuracy: ',
                          np.sum(np.round(test_acts[-1]) == self.test_y) * 100 / self.test_y.shape[1])


# Convolutional Neural Network, inheriting all methods and constants from ANN
class CNN(ANN):
    filter_size = []
    num_channels = []
    padding = []
    stride = []
    pooling = []
    pool_size = []
    pool_stride = []
    act_conv = []
    num_conv_layers = 0
    conv_weights = []
    conv_biases = []
    conv_dimension = []
    final_conv_dimension =

    def __init__(self, filter_size, num_channels, padding, stride, pooling, pool_size, pool_stride, fc_dims, learning_rate, train_x, train_y,
                 act_conv, act_hid, act_out, cost, show_error=0, test_x=[], test_y=[]):
        # Convolution layer parameters
        self.filter_size = filter_size          # Must include the dimensions of the input data
        self.padding = padding
        self.stride = stride
        self.pooling = pooling
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.act_conv = act_conv
        self.num_channels = num_channels        # Must include number of channels for input data
        self.num_conv_layers = len(filter_size)

        # Calculate dimensions of the final convolution layer
        i_dim = train_x.shape[2]
        j_dim = train_x.shape[3]
        for c_layer in range(1, self.num_conv_layers):
            i_dim = int((int((i_dim + 2 * padding[c_layer] - filter_size[c_layer]) / stride[c_layer]) + 1 - pool_size[c_layer]) / pool_stride[c_layer]) + 1
            j_dim = int((int((j_dim + 2 * padding[c_layer] - filter_size[c_layer]) / stride[c_layer]) + 1 - pool_size[c_layer]) / pool_stride[c_layer]) + 1

        # Fully connected layer and general parameters
        self.num_layers = len(fc_dims) + 1                      # Includes the final convolution layer, and treats it like a standard input layer
        self.layer_dims = fc_dims
        self.layer_dims.insert(0, i_dim * j_dim * num_channels[-1])
        self.lr_initial = learning_rate
        self.show_error = show_error
        self.train_x = train_x          # Must be of .shape = (# of training examples, # of channels in each example, height of each example, width of each example)
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.act_hid = act_hid
        self.act_out = act_out
        self.cost = cost

        self.reset()

    def reset(self):
        # Convolution layers parameters initialization
        self.conv_weights = [None] + [np.random.randn(self.num_channels[cl], self.num_channels[cl - 1],
                                                      self.filter_size[cl], self.filter_size[cl]) *
                                      np.sqrt(2 / (self.num_channels[cl - 1] * self.filter_size[cl] ** 2))  # Modify the initializaiton based upon the number of values used during each operation:
                                      for cl in range(1, self.num_conv_layers)]                             # number of parameters per filter multiplied by the number of channels
        self.conv_biases = [None] + [np.zeros((self.num_channels[cl], 1)) for cl in range(1, self.num_conv_layers)]
        self.weights = [None] + [np.random.randn(self.layer_dims[layer], self.layer_dims[layer - 1]) *
                                 np.sqrt(2 / self.layer_dims[layer - 1]) for layer in range(1, self.num_layers)]
        self.biases = [None] + [np.zeros((self.layer_dims[layer], 1)) for layer in range(1, self.num_layers)]

    def pooling_step(self, pre_pool, c_layer):
        i_post_pool_size = int((pre_pool.shape[2] - self.pool_size[c_layer]) / self.pool_stride[c_layer]) + 1
        j_post_pool_size = int((pre_pool.shape[3] - self.pool_size[c_layer]) / self.pool_stride[c_layer]) + 1
        post_pool = np.zeros((pre_pool.shape[0], pre_pool.shape[1], i_post_pool_size, j_post_pool_size))

        if self.pooling[c_layer] == 0:
            return pre_pool

        for i in range(0, i_post_pool_size):
            for j in range(0, j_post_pool_size):
                ipos_start = i * self.pool_stride[c_layer]
                jpos_start = j * self.pool_stride[c_layer]
                if self.pooling[c_layer] == 'max':
                    post_pool[:, :, i, j] = np.amax(pre_pool[:, :, ipos_start:ipos_start + self.pool_size[c_layer], jpos_start:jpos_start + self.pool_size[c_layer]], axis=(2, 3))
                elif self.pooling[c_layer] == 'average':
                    post_pool[:, :, i, j] = np.sum(pre_pool[:, :, ipos_start:ipos_start + self.pool_size[c_layer], jpos_start:jpos_start + self.pool_size[c_layer]], axis=(2, 3)) / (self.pool_size[c_layer] ** 2)

        return post_pool

    def forward_propagation(self, input_data):
        acts = [[None] for i in range(self.num_layers)]
        z_values = [[None] for i in range(self.num_layers)]
        acts[0] = input_data

        for layer in range(1, self.num_layers):
            z_values[layer] = np.dot(self.weights[layer], acts[layer - 1]) + self.biases[layer]
            if layer == (self.num_layers - 1):
                acts[layer] = self.act_out.func(z_values[layer])
            else:
                acts[layer] = self.act_hid.func(z_values[layer])
        return z_values, acts

    # Forward propagation function
    def conv_forward_propagation(self, input_data):
        conv_acts = [[None] for i in range(self.num_conv_layers)]
        conv_acts[0] = input_data

        for c_layer in range(1, self.num_conv_layers):
            i_post_filter_size = int((conv_acts[c_layer - 1].shape[2] + 2 * self.padding[c_layer] - self.filter_size[c_layer]) / self.stride[c_layer]) + 1
            j_post_filter_size = int((conv_acts[c_layer - 1].shape[3] + 2 * self.padding[c_layer] - self.filter_size[c_layer]) / self.stride[c_layer]) + 1
            temp = np.zeros((conv_acts[c_layer-1].shape[0], self.num_channels[c_layer], i_post_filter_size, j_post_filter_size))
            padded_prev_act = np.pad(conv_acts[c_layer - 1], ((0, 0), (0, 0), (self.padding[c_layer], self.padding[c_layer]), (self.padding[c_layer], self.padding[c_layer])))
            for channel in range(0, self.num_channels[c_layer]):
                for i in range(0, i_post_filter_size):
                    for j in range(0, j_post_filter_size):
                        ipos_start = self.stride[c_layer] * i
                        jpos_start = self.stride[c_layer] * j
                        temp[:, channel, i, j] = np.sum(self.conv_weights[c_layer][channel] * padded_prev_act[:, :, ipos_start:ipos_start + self.filter_size[c_layer],
                                                                                    jpos_start:jpos_start + self.filter_size[c_layer]], axis=(1, 2, 3)) + self.conv_biases[c_layer][channel]
            conv_acts[c_layer] = self.pooling_step(self.act_conv.func(temp), c_layer)
        flattened = conv_acts[-1].reshape((-1, 1), order='C')
        z_values, acts = self.forward_propagation(flattened)

        return conv_acts, z_values, acts
