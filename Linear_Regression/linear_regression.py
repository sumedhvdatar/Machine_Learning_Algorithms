import numpy as np
import itertools
#Source : https://www.cs.toronto.edu/~frossard/post/linear_regression/
def squared_l2_norm(w):
    from numpy import linalg as LA
    """
    Implements the squared L2 norm for weight regularization. ||W||^2
    :param w: column vector of weights [n, 1]
    :return: squared l2 norm of w
    """
    l2 =  np.sum(np.square(w))
    return l2
    # raise Warning("You must implement squared_l2_norm!")
    # return 0

def mean_squared_error(y_hat, y):
    """
    Implements the mean squared error cost function for linear regression
    :param y_hat: predicted values (model output), vector [n, 1]
    :param y: target values vector [n,1]
    :return: mean squared error (scalar)
    Ref : https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy
    """
    mse = (np.square(y_hat.T - y.T)).mean(axis=1)
    # raise Warning("You must implement mean_squared_error!")
    return mse[0]

def calculate_batches(X, batch_size):
    """
    Already implemented, don't worry about it!
    :param X:
    :param batch_size:
    :return:
    """
    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)
    args = [iter(indices)] * batch_size
    batch_indices = itertools.zip_longest(*args, fillvalue=None)
    return [list(j for j in i if j is not None) for i in batch_indices]

class LinearRegression(object):
    def __init__(self, input_dimensions=2, seed=1234):
        """
        Initialize a linear regression model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        """
        np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of a linear regression model, initalize using random numbers.
        """
        self.weights = np.random.randn(3,1)
        # print(self.weights)
        # raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Use 'the bias trick for this assignment'")

    def fit(self, X_train, y_train, X_val, y_val, num_epochs=10, batch_size=16, alpha=0.01, _lambda=0.0):
        """
        Stochastic Gradient Descent training loop for a linear regression model
        :param X_train: Training input data [n_samples, n_features+1], assume last column is 1's for the bias
        :param y_train: Training target values [n_samples, 1]
        :param X_val: Validation input data [n_samples, n_features+1], assume last column is 1's for the bias
        :param y_val: Validation target values [n_samples, 1]
        :param num_epochs: Number of training epochs (default 10)
        :param batch_size: Number of samples in a single batch (default 16)
        :param alpha: Learning rate for gradient descent update (default 0.01)
        :param _lambda: Coefficient for L2 weight regularization (default 0.0)
        :return: (train_error_, val_error) two arrays of MSE values for each epoch (one for training and validation error)
        """
        # raise Warning("You must implement fit! This function should implement a mini-batch stochastic gradient descent training loop")

        train_error = [] # append your MSE on training set to this after each epoch
        val_error = [] # append your MSE on validation set to this after each epoch

        batch_indices = calculate_batches(X_train, batch_size)
        for epoch in range(num_epochs):
            for batch in batch_indices:
                self._train_on_batch(X_train[batch], y_train[batch], alpha, _lambda)
                y_estimate = X_train[batch].dot(self.weights).flatten()
                mse = mean_squared_error(y_estimate,y_train[batch])
                train_error.append(mse)

                y_estimate = X_val.dot(self.weights).flatten()
                # print(y_estimate.shape)
                # print(y_val[batch].shape)
                loss = mean_squared_error(y_estimate, y_val)
                val_error.append(loss)
            # calculate error on validation set here
        # for epoch in range(num_epochs):
        #     for batch in batch_indices:
        #         self._train_on_batch(X_val[batch], y_val[batch], alpha, _lambda)
        #         y_estimate = X_val[batch].dot(self.weights).flatten()
        #         print(y_estimate.shape,y_val[batch].shape)
        #         mse = mean_squared_error(y_estimate,y_val[batch])
        #         val_error.append(mse)

        # return: two lists (train_error_after_each epoch, validation_error_after_each_epoch)
        return train_error, val_error

    def predict(self, X):
        """
        Make a prediction on an array of inputs, must already contain bias as last column
        :param X: Array of input [n_samples, n_features+1]
        :return: Array of model outputs [n_samples, 1]
        """
        #The assumption here is that bias is equal to 0 else it will not match with the output given in the test.
        b = np.zeros(self.input_dimensions)
        w = self.weights
        predicted_value = np.dot(X,w)
        return predicted_value
        # raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")

    def _train_on_batch(self, X, y, alpha, _lambda):
        """
        Given a single batch of data, and the necessary hyperparameters, perform a single batch gradient update. This function should update the model weights.
        :param X: Batch of training input data [batch_size, n_features+1]
        :param y: Batch of training targets [batch_size, 1]
        :param alpha: Learning rate (scalar i.e. 0.01)
        :param _lambda: Regularization strength coefficient (scalar i.e. 0.0001)
        """
        # calculate output
        # calculate errors, mean squared error, and squared L2 regularization
        # calculate gradients of cross entropy and L2  w.r.t weights
        # perform gradient descent update
        # Note: please make use of the functions _mse_gradient and _l2_regularization_gradient
        # raise Warning("You must implement train on batch. This function should perform a stochastic gradient descent update on a single batch of samples")
        gradient = self._mse_gradient(X,y)
        # y_estimate = X.dot(self.weights).flatten()
        # mse = mean_squared_error(y_estimate,y)
        new_weights = self.weights - (alpha * gradient) - (alpha * _lambda * self._l2_regularization_gradient())
        self.weights = new_weights
        # return new_weights


    def _mse_gradient(self, X, y):
        """
        Compute gradient of MSE objective w.r.t model weights.
        :param X: Set of input data [n_samples, n_features+1]
        :param y: Set of target values [n_samples, 1]
        :return: Gradient of MSE w.r.t model weights [n_features+1, 1]
        """
        # implement the mean squared error gradient for a linear regression model
        y_estimate = X.dot(self.weights).flatten()
        error = (y.flatten() - y_estimate)
        mse = mean_squared_error(y_estimate,y)
        gradient = np.asarray(-(1.0 / len(X)) * error.dot(X))
        return np.float32([gradient]).T
        # raise Warning("You must implement the gradient. Do not include alpha in your calculation. Gradient should be same dimension as your weights")

    def _l2_regularization_gradient(self):
        """
        Compute gradient for l2 weight regularization
        :return: Gradient of squared l2 norm w.r.t model weights [n_features+1, 1]
        """
        return self.weights
        #raise Warning("You must implement the gradient for the squared l2 norm of the model weights. Do not include lambda in this part of the calculation")

if __name__ == "__main__":
    print("This is library code. You may implement some functions here to do your own debugging if you want, but they won't be graded")
    #----------------------------------------------
    #Debugging for MSE
    # outputs = np.float32([[1, 2, 3, 4]]).T
    # targets = np.float32([[1.1, 2.2, 3.3, 4.4]]).T
    # actual = mean_squared_error(outputs, targets)
    #------------------------------------------------

    #-----------------------------------------------
    #debugging for L2 squared
    # w = np.float32([[1, 2, 3, 4]]).T
    # actual = squared_l2_norm(w)
    # desired = 30.00000225043209
    #-----------------------------------------------

    #debugging for weight dimension
    # model = LinearRegression(input_dimensions=2)
    # w1 = model._initialize_weights()

    #debugging with predict function
    # model = LinearRegression(input_dimensions=2)
    # model.weights = np.float32([[1, 2, 4]]).T
    # X = np.float32([[1, 2, 1],
    #                 [0, 0, 1]])
    # desired = np.float32([[9, 4]]).T
    # actual = model.predict(X)
    # print(actual)

    #debugging for gradients
    # model = LinearRegression(input_dimensions=2)
    # model.weights = np.float32([[1, 2, 4]]).T
    # X = np.float32([[1, 2, 1],
    #                 [0, 0, 1]])
    # y = np.float32([[10, 2]]).T
    # gradient = model._mse_gradient(X, y)
    # print(gradient)
    # desired = np.float32([[-0.5, -1., 0.5]]).T
    # print(desired.shape)

    #debugg check weight input dimension
    # model = LinearRegression(input_dimensions=2)
    # print(model.weights.shape[0])

    #debugging train on batch
    # model = LinearRegression(input_dimensions=2)
    # weights_old = np.float32([[1, 2, 4]]).T
    # model.weights = np.float32([[1, 2, 4]]).T

    # X = np.float32([[1, 2, 1],
    #                 [0, 0, 1]])
    # y = np.float32([[10, 2]]).T
    # print(model.weights)
    # model._train_on_batch(X, y, 0.3, _lambda=0.001)
    # desired = np.float32([[-0.14970, -0.29940, 0.15120]]).T
    # weight_delta = (weights_old - model.weights)
    # print(weight_delta)
    # print(desired)


    # # debug fit functionality
    # import sklearn.model_selection
    # import numpy as np

    # from linear_regression import LinearRegression

    # X = np.zeros((900, 3), dtype=np.float32)
    # num_samples = 30

    # xx = np.linspace(-5, 5, num_samples)
    # XX, YY = np.meshgrid(xx, xx)
    # X[:, 0] = XX.flatten()
    # X[:, 1] = YY.flatten()
    # X[:, -1] = 1  # a column of 1's for the bias trick
    # Z = 0.1 * XX + 0.2 * YY + 0.4
    # y = Z.reshape(-1, 1)
    # X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y)
    # model = LinearRegression(input_dimensions=2)
    # train_mse, val_mse = model.fit(X_train, y_train, X_val, y_val, num_epochs=20, batch_size=4, alpha=0.1, _lambda=0.0)
    # # final_train_mse = train_mse[-1]
    # desired_weights = np.float32([[0.1, 0.2, 0.4]]).T
    # print(model.weights)
    # np.testing.assert_allclose(model.weights, desired_weights, rtol=1e-3, atol=1e-3)
    # assert final_train_mse < 0.001
    # assert final_train_mse < 0.00001
    # assert final_train_mse < 1e-10