#Author : Sumedh Vilas Datar
import numpy as np
import itertools

def squared_l2_norm(w):
    """
    Implements the squared L2 norm for weight regularization. ||W||^2
    :param w: column vector of weights [n, 1]
    :return: squared l2 norm of w
    """
    l2 = np.sum(np.square(w))
    return l2
    # raise Warning("You must implement squared_l2_norm! This is for calculating regularization")
    # return 0

def user_log(val):
    val = val + 0.00000001
    return val


def binary_cross_entropy(y_hat, y):
    """
    Implements the binary cross-entropy loss function for logistic regression
    :param y_hat: predicted values (model output), vector [n, 1]
    :param y: target values vector [n,1], binary values either 0 or 1
    :return: binary cross-entropy loss between y and y_hat
    """
    cost = -(np.dot(y.T,np.log(y_hat + 1e-8)) + np.dot((1-y).T,np.log(1-y_hat + 1e-8)))
    return cost

def sigmoid(x):
    """
    Compute sigmoid function on x, elementwise
    :param x: array of inputs [m, n]
    :return: array of outputs [m, n]
    """
    sigm = 1. / (1. + np.exp(-x))
    # raise Warning("You must implement sigmoid!")
    return sigm

def accuracy(y_pred, y):
    """
    Compute accuracy of predictions y_pred based on ground truth values y
    :param y_pred: Predicted values, THRESHOLDED TO 0 or 1, not probabilities
    :param y: Ground truth (target) values, also 0 or 1
    :return: Accuracy (scalar) 0.0 to 1.0
    """
    total = len(y_pred)
    correct_prediction = 0
    for i in range(0,len(y_pred)):
        if y_pred[i] == y[i]:
            correct_prediction = correct_prediction + 1
    percentage_of_correct = (correct_prediction/total)
    return percentage_of_correct


def calculate_batches(X, batch_size):
    """
    Already implemented, don't worry about it
    :param X:
    :param batch_size:
    :return:
    """
    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)
    args = [iter(indices)] * batch_size
    batch_indices = itertools.zip_longest(*args, fillvalue=None)
    return [list(j for j in i if j is not None) for i in batch_indices]

class LogisticRegression(object):
    def __init__(self, input_dimensions=2, seed=1234):
        """
        Initialize a Logistic Regression model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param seed: Random seed for controlling/repeating experiments
        """
        np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights of a logistic regression model using random numbers
        """
        self.weights = np.random.randn(3,1)
        # raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Use 'the bias trick' for this assignment")

    def fit(self, X_train, y_train, X_val, y_val, num_epochs=10,batch_size=4, alpha=0.01, _lambda=0.0):
        """
        Stochastic Gradient Descent training loop for a logistic regression model
        :param X_train: Training input data [n_samples, n_features+1], assume last column is 1's for the bias
        :param y_train: Training target values [n_samples, 1]
        :param X_val: Validation input data [n_samples, n_features+1], assume last column is 1's for the bias
        :param y_val: Validation target values [n_samples, 1]
        :param num_epochs: Number of training epochs (default 10)
        :param batch_size: Number of samples in a single batch (default 16)
        :param alpha: Learning rate for gradient descent update (default 0.01)
        :param _lambda: Coefficient for L2 weight regularization (default 0.0)
        :return: (train_error_, val_error) two arrays of cross-entropy values for each epoch (one for training and validation error)
        """
        # raise Warning("You must implement fit! This function should implement a mini-batch stochastic gradient descent training loop")
        train_xent = []  # append your cross-entropy on training set to this after each epoch
        val_xent = []  # append your cross-entropy on validation set to this after each epoch

        batch_indices = calculate_batches(X_train, batch_size)

        for epoch in range(num_epochs):
            for batch in batch_indices:
                self._train_on_batch(X_train[batch], y_train[batch], alpha, _lambda)
                y_estimate = self.predict_proba(X_train[batch])
                # print(y_estimate.shape)
                # print(y_train[batch].shape)
                loss = binary_cross_entropy(y_estimate,y_train[batch])
                train_xent.append(loss)

                y_estimate = self.predict_proba(X_val)
                # print(y_estimate.shape)
                # print(y_val[batch].shape)
                loss = binary_cross_entropy(y_estimate, y_val)
                val_xent.append(loss)
        return (train_xent, val_xent)

    def predict_proba(self, X):
        """
        Make a prediction on an array of inputs, must already contain bias as last column
        :param X: Array of input [n_samples, n_features+1]
        :return: Array of model outputs [n_samples, 1]. Each entry is a probability between 0 and 1
        """
        probability = sigmoid(np.dot(X,self.weights))
        return probability
        # raise Warning("You must implement predict_proba. This function should make a prediction on a batch (matrix) of inputs. The output should be probabilities.")

    def predict(self, X):
        """
        Make a prediction on an array of inputs, and choose the nearest class, must already contain bias as last column
        :param X: Array of input [n_samples, n_features+1]
        :return: Array of model outputs [n_samples, 1]. Each entry is class ID (0 or 1)
        """
        probability = self.predict_proba(X)
        result = []
        for p in probability:
            if p >= 0.5:
                result.append(1)
            elif p < 0.5:
                result.append(0)
        result = np.asarray([result])
        return result.T

        # raise Warning("You must implement predict. This function should make a prediction on a batch (matrix) of inputs. The output should be class ID (0 or 1)")

    def _train_on_batch(self, X, y, alpha, _lambda):
        """
        Given a single batch of data, and the necessary hyperparameters, perform a single batch gradient update. This function should update the model weights.
        :param X: Batch of training input data [batch_size, n_features+1]
        :param y: Batch of training targets [batch_size, 1]
        :param alpha: Learning rate (scalar i.e. 0.01)
        :param _lambda: Regularization strength coefficient (scalar i.e. 0.0001)
        """
        # calculate output
        # calculate errors, binary cross entropy, and squared L2 regularization
        # calculate gradients of cross entropy and L2  w.r.t weights
        # perform gradient descent update
        # raise Warning("You must implement train on batch. This function should perform a stochastic gradient descent update on a single batch of samples")
        gradient = self._binary_cross_entropy_gradient(X, y)
        # y_estimate = X.dot(self.weights).flatten()
        # mse = mean_squared_error(y_estimate,y)
        new_weights = self.weights - (alpha * gradient) - (alpha * _lambda * self._l2_regularization_gradient())
        self.weights = new_weights

    def _binary_cross_entropy_gradient(self, X, y):
        """
        Compute gradient of binary cross-entropy objective w.r.t model weights.
        :param X: Set of input data [n_samples, n_features+1]
        :param y: Set of target values [n_samples, 1]
        :return: Gradient of cross-entropy w.r.t model weights [n_features+1, 1]
        """
        # implement the binary cross-entropy gradient for logistic regression
        # this is the gradient W.R.T the weights
        # raise Warning("You must implement the binary cross entropy gradient")
        y_hat = sigmoid(np.dot(X,self.weights))
        gradient = np.dot(X.T, (y_hat - y)) / y.shape[0]
        # print(gradient)
        return gradient

    def _l2_regularization_gradient(self):
        """
        Compute gradient for l2 weight regularization
        :return: Gradient of squared l2 norm w.r.t model weights [n_features+1, 1]
        """
        return self.weights
        # raise Warning("You must implement the gradient for the squared l2 norm ")

if __name__ == "__main__":
    print("This is library code. You may implement some functions here to do your own debugging if you want, but they won't be graded")
    #debug accuracy
    # y_pred = np.float32([[0, 1, 1, 0, 1]]).T
    # y = np.float32([[1, 0, 1, 0, 1]]).T
    # actual = accuracy(y_pred, y)
    # desired = 3. / 5.
    # np.testing.assert_allclose(actual, desired)

    # debug binary cross entropy
    # outputs = np.float32([[0.0, 0.1, 0.9, 0.8]]).T
    # targets = np.float32([[1, 0, 1, 0]]).T
    # actual = binary_cross_entropy(outputs, targets)
    # print(actual)
    # # print("cool")
    #
    # outputs = np.float32([[1, 0, 1, 0]]).T
    # targets = np.float32([[1, 0, 1, 0]]).T
    # actual = binary_cross_entropy(outputs, targets)
    # print(actual)
    # print("cool")
    #debug predict proba
    # model = LogisticRegression(input_dimensions=2)
    # model.weights = np.float32([[1, 2, 4]]).T
    # X = np.float32([[1, 2, 1],
    #                 [0, 0, -2]])
    # desired = np.float32([[0.9987, 0.0003]]).T
    # print(desired)
    # actual = model.predict_proba(X)
    # print(actual)

    #debug predict
    # model = LogisticRegression(input_dimensions=2)
    # model.weights = np.float32([[1, 2, 4]]).T
    # X = np.float32([[1, 2, 1],
    #                 [0, 0, -2]])
    # desired = np.float32([[1, 0]]).T
    # print(desired)
    # actual = model.predict(X)
    # print(actual)

    #Debug fit functionality
    # import sklearn.model_selection
    # import sklearn.datasets
    # import numpy as np
    # from logistic_regression import LogisticRegression, accuracy

    # X = np.zeros((1000, 3), dtype=np.float32)
    # X[:, -1] = 1
    # features, targets = sklearn.datasets.make_blobs(1000, 2, 2, cluster_std=0.3)
    # X[:, [0, 1]] = features
    # y = targets[:, np.newaxis]

    # X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y)
    # model = LogisticRegression(input_dimensions=2)
    # train_xent, val_xent = model.fit(X_train, y_train, X_val, y_val, num_epochs=20, batch_size=4, alpha=0.1,
    #                                  _lambda=0.0)
    # predictions = model.predict(X_val)
    # print(accuracy(predictions,y_val))
    # assert accuracy(predictions, y_val) >= 0.65
    # assert accuracy(predictions, y_val) >= 0.90
    # assert accuracy(predictions, y_val) >= 0.99

    #Debug Train on Batch
    # from logistic_regression import LogisticRegression
    #
    # model = LogisticRegression(input_dimensions=2)
    # weights_old = np.float32([[1, 2, 4]]).T
    # model.weights = np.float32([[1, 2, 4]]).T
    #
    # X = np.float32([[1, 2, 1],
    #                 [0, 0, 1]])
    # y = np.float32([[1, 0]]).T
    # model._train_on_batch(X, y, 0.3, _lambda=0.001)
    # desired = np.float32([[0.000281, 0.000563, 0.14848]]).T
    # weight_delta = (weights_old - model.weights)
    # print(weight_delta)
    # print("done")

